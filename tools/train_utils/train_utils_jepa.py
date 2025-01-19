# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist


def load_checkpoint(
        path,
        optimizer,
        scaler,
        encoder,
        predictor,
        target_encoder,
        logger,
        map_predictor=None,
):
    # try:
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    encoder.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained context encoder from epoch {epoch}')

    # -- loading predictor
    pretrained_dict = checkpoint['predictor']
    predictor.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained predictor from epoch {epoch}')

    # -- loading target_encoder
    if target_encoder is not None:
        pretrained_dict = checkpoint['target_encoder']
        target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained target encoder from epoch {epoch}')
    
    if map_predictor is not None:
        pretrained_dict = checkpoint['map_predictor']
        map_predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained map predictor from epoch {epoch}')

    # -- loading optimizer
    optimizer.load_state_dict(checkpoint['opt'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {path}')
    del checkpoint

    # except Exception as e:
    #     logger.info(f'Encountered exception when loading checkpoint {e}')
    #     epoch = 0
    
    return encoder, predictor, target_encoder, map_predictor, optimizer, scaler, epoch


def save_checkpoint(
        encoder,
        predictor,
        target_encoder,
        map_predictor,
        optimizer,
        scaler,
        epoch,
        path
):
    save_dict = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'map_predictor': map_predictor.state_dict() if map_predictor is not None else None,
        'opt': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
    }
    torch.save(save_dict, path)


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def get_jepa_loss(predicted_encodings, target_encodings, mse_coeff, std_coeff, cov_coeff):
    num_center_objects, d_model = predicted_encodings.shape
    
    # MSE loss
    mse_loss = torch.nn.functional.smooth_l1_loss(predicted_encodings, target_encodings)
    #mse_loss = AllReduce.apply(mse_loss)

    # Variance loss
    # Turn encoded features into [num_center_objects, d_model]
    #output_encoder = output_encoder - torch.mean(output_encoder, dim=0)
    #output_target_encoder = output_target_encoder - torch.mean(output_target_encoder, dim=0)
    std_encoder = torch.sqrt(predicted_encodings.var(dim=0) + 0.0001)
    #std_target_encoder = torch.sqrt(output_target_encoder.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.nn.functional.relu(1 - std_encoder)) / 2 #+ torch.mean(torch.nn.functional.relu(2 - std_target_encoder)) / 2
    #std_loss = AllReduce.apply(std_loss)

    # Covariance loss
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    cov_encoder = (predicted_encodings.T @ predicted_encodings) / (num_center_objects - 1)
    #cov_target_encoder = (output_target_encoder.T @ output_target_encoder) / (num_center_objects - 1)
    cov_loss = off_diagonal(cov_encoder).pow_(2).sum().div(d_model) #+ off_diagonal(cov_target_encoder).pow_(2).sum().div(d_model)
    #cov_loss = AllReduce.apply(cov_loss)

    # Weighted loss
    loss = (mse_coeff * mse_loss + std_coeff * std_loss + cov_coeff * cov_loss)
    return loss, (mse_coeff * mse_loss, std_coeff * std_loss, cov_coeff * cov_loss)


def get_jepa_loss_with_map(predicted_encodings, target_encodings, predicted_map_encodings, map_target_encodings, mse_coeff=1.0, std_coeff=1.0, cov_coeff=0.04):
    num_center_objects, d_model = predicted_encodings.shape
    # MSE loss
    mse_loss = torch.nn.functional.smooth_l1_loss(predicted_encodings, target_encodings)
    map_mse_loss = torch.nn.functional.smooth_l1_loss(predicted_map_encodings, map_target_encodings)
    #mse_loss = (mse_loss + map_mse_loss) / 2
    #mse_loss = AllReduce.apply(mse_loss)

    # Variance loss
    # Turn encoded features into [num_center_objects, d_model]
    #output_encoder = output_encoder - torch.mean(output_encoder, dim=0)
    #output_target_encoder = output_target_encoder - torch.mean(output_target_encoder, dim=0)
    std_encoder = torch.sqrt(predicted_encodings.var(dim=0) + 0.0001)
    #std_target_encoder = torch.sqrt(output_target_encoder.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.nn.functional.relu(1 - std_encoder)) / 2 #+ torch.mean(torch.nn.functional.relu(2 - std_target_encoder)) / 2
    #std_loss = AllReduce.apply(std_loss)

    # Covariance loss
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    cov_encoder = (predicted_encodings.T @ predicted_encodings) / (num_center_objects - 1)
    #cov_target_encoder = (output_target_encoder.T @ output_target_encoder) / (num_center_objects - 1)
    cov_loss = off_diagonal(cov_encoder).pow_(2).sum().div(d_model) #+ off_diagonal(cov_target_encoder).pow_(2).sum().div(d_model)
    #cov_loss = AllReduce.apply(cov_loss)

    # Weighted loss
    loss = (mse_coeff * mse_loss + std_coeff * std_loss + cov_coeff * cov_loss + mse_coeff * map_mse_loss)
    return loss, (mse_coeff * mse_loss, std_coeff * std_loss, cov_coeff * cov_loss, mse_coeff * map_mse_loss)


def train_one_epoch(context_encoder, predictor, target_encoder, map_predictor, optimizer, train_loader, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, scheduler=None, show_grad_curve=False,
                    logger=None, logger_iter_interval=50, cur_epoch=None, total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, momentum_scheduler=None, scaler=None, cfg=None, wd_scheduler=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    optimizer, optimizer_2 = optimizer if isinstance(optimizer, list) else (optimizer, None)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()
        if optimizer_2 is not None:
            optimizer_2.zero_grad()

        ################## JEPA FORWARD ##################
        batch = context_encoder(batch)
        predicted_obj_features = predictor(batch['center_objects_feature'])
        if map_predictor is not None:
            predicted_map_features = map_predictor(batch['map_feature'])
        with torch.no_grad():
            target_encoding, target_map_encoding = target_encoder(batch, target=True)
        if map_predictor is not None:
            loss, single_losses = get_jepa_loss_with_map(predicted_obj_features,
                                                         target_encoding,
                                                         predicted_map_features,
                                                         target_map_encoding,
                                                         mse_coeff=cfg.MODEL.CONTEXT_ENCODER.mse_coeff,
                                                         std_coeff=cfg.MODEL.CONTEXT_ENCODER.std_coeff,
                                                         cov_coeff=cfg.MODEL.CONTEXT_ENCODER.cov_coeff
                                                         )
        else:
            loss, single_losses = get_jepa_loss(predicted_obj_features, target_encoding, mse_coeff=cfg.MODEL.CONTEXT_ENCODER.mse_coeff, std_coeff=cfg.MODEL.CONTEXT_ENCODER.std_coeff, cov_coeff=cfg.MODEL.CONTEXT_ENCODER.cov_coeff)
            
        loss = AllReduce.apply(loss)

        if scaler is not None:
            scaler.scale(loss).backward()
            total_norm = clip_grad_norm_(context_encoder.parameters(), optim_cfg.GRAD_NORM_CLIP)
            total_norm_pred = clip_grad_norm_(predictor.parameters(), optim_cfg.GRAD_NORM_CLIP)
            if map_predictor is not None:
                total_norm_map = clip_grad_norm_(map_predictor.parameters(), optim_cfg.GRAD_NORM_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            total_norm = clip_grad_norm_(context_encoder.parameters(), optim_cfg.GRAD_NORM_CLIP)
            total_norm_pred = clip_grad_norm_(predictor.parameters(), optim_cfg.GRAD_NORM_CLIP)
            if map_predictor is not None:
                total_norm_map = clip_grad_norm_(map_predictor.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

        if optimizer_2 is not None:
            optimizer_2.step()
        
        if scheduler is not None:
            scheduler.step()
        if wd_scheduler is not None:
            wd_scheduler.step()
        
        # JEPA specific update of the target_encoder params
        with torch.no_grad():
            # m = next(momentum_scheduler)
            m = next(momentum_scheduler)
            for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        accumulated_iter += 1
        tb_dict = {'loss': loss.item(), 'lr': cur_lr}
        disp_dict = {'loss': loss.item(), 'lr': cur_lr}
        if map_predictor is not None:
            tb_dict.update({'mse_loss': single_losses[0].item(), 'std_loss': single_losses[1].item(), 'cov_loss': single_losses[2].item(), 'map_mse_loss': single_losses[3].item()})
        else:
            tb_dict.update({'mse_loss': single_losses[0].item(), 'std_loss': single_losses[1].item(), 'cov_loss': single_losses[2].item()})

        # log to console and tensorboard
        if rank == 0:
            if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                trained_time_past_all = tbar.format_dict['elapsed']
                second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                trained_time_each_epoch = pbar.format_dict['elapsed']
                remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                disp_str += f', lr={disp_dict["lr"]}'
                batch_size = batch.get('batch_size', None)
                logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                            f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                            f'{disp_str}')

            if tb_log is not None:
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    if 'embeddings' not in key and 'object_ids' not in key:
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                    else:
                        continue
                if (accumulated_iter % total_it_each_epoch == 0 and optim_cfg.get('JEPA', False)) and optim_cfg.get('LOG_EMBEDDINGS', False):
                    tb_log.add_embedding(tb_dict['context_embeddings'], metadata=tb_dict['object_ids'], global_step=accumulated_iter, tag='context_embeddings')
                    tb_log.add_embedding(tb_dict['predicted_embeddings'], metadata=tb_dict['object_ids'], global_step=accumulated_iter, tag='predicted_embeddings')
                    tb_log.add_embedding(tb_dict['target_embeddings'], metadata=tb_dict['object_ids'], global_step=accumulated_iter, tag='target_embeddings')
                tb_log.add_scalar('train/total_norm', total_norm, accumulated_iter)

            time_past_this_epoch = pbar.format_dict['elapsed']
            # if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
            #     ckpt_name = ckpt_save_dir / 'latest_model'
            #     save_checkpoint(
            #         checkpoint_state(model, optimizer, cur_epoch, accumulated_iter, scheduler=scheduler), filename=ckpt_name,
            #     )
            #     logger.info(f'Save latest model to {ckpt_name}')
            #     ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def learning_rate_decay(i_epoch, optimizer, optim_cfg):
    if isinstance(optimizer, list):
        optimizer, optimizer_2 = optimizer

    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.3

    if optim_cfg.OPTIMIZER == 'complete_traj':
        if i_epoch > 0 and i_epoch % 5 == 0:
            for p in optimizer_2.param_groups:
                p['lr'] *= 0.3


def train_model(context_encoder, predictor, target_encoder, map_predictor, optimizer, scaler, scheduler, wd_scheduler,
                train_loader, optim_cfg, train_sampler, momentum_scheduler,
                start_epoch, total_epochs, start_iter, rank, ckpt_save_dir,
                ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, tb_log=None,
                test_loader=None, logger=None, eval_output_dir=None, cfg=None, dist_train=False,
                logger_iter_interval=50, ckpt_save_time_interval=300, show_grad_curve=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
    
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            torch.cuda.empty_cache()
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            accumulated_iter = train_one_epoch(
                context_encoder, predictor, target_encoder, map_predictor, optimizer, train_loader,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                scheduler=scheduler, cur_epoch=cur_epoch, total_epochs=total_epochs,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, momentum_scheduler=momentum_scheduler, scaler=scaler,
                show_grad_curve=show_grad_curve, cfg=cfg, wd_scheduler=wd_scheduler
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d.pth' % trained_epoch)
                save_checkpoint(
                    context_encoder,
                    predictor,
                    target_encoder,
                    map_predictor,
                    optimizer,
                    scaler,
                    trained_epoch,
                    ckpt_name
                )

            # eval the model
            if test_loader is not None and (trained_epoch % ckpt_save_interval == 0): #or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10):
                from eval_utils.eval_utils_jepa import eval_one_epoch
                torch.cuda.empty_cache()
                tb_dict = eval_one_epoch(
                    cfg, context_encoder,
                    predictor,
                    target_encoder,
                    map_predictor, test_loader, epoch_id=trained_epoch, logger=logger, dist_test=dist_train,
                    result_dir=eval_output_dir, save_to_file=False, logger_iter_interval=max(logger_iter_interval // 5, 1)
                )
                if cfg.LOCAL_RANK == 0:
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('eval/' + key, val, trained_epoch)


# def model_state_to_cpu(model_state):
#     model_state_cpu = type(model_state)()  # ordered dict
#     for key, val in model_state.items():
#         model_state_cpu[key] = val.cpu()
#     return model_state_cpu


# def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, scheduler=None):
#     optim_state = optimizer.state_dict() if optimizer is not None else None
#     try:
#         scheduler_state = scheduler.state_dict() if scheduler is not None else None
#     except:
#         scheduler_state = None
#     if model is not None:
#         if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#             model_state = model_state_to_cpu(model.module.state_dict())
#         else:
#             model_state = model.state_dict()
#     else:
#         model_state = None

#     try:
#         import mtr
#         version = 'mtr+' + mtr.__version__
#     except:
#         version = 'none'

#     return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version, 'scheduler_state': scheduler_state}


# def save_checkpoint(state, filename='checkpoint'):
#     if False and 'optimizer_state' in state:
#         optimizer_state = state['optimizer_state']
#         state.pop('optimizer_state', None)
#         optimizer_filename = '{}_optim.pth'.format(filename)
#         torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

#     filename = '{}.pth'.format(filename)
#     torch.save(state, filename)
