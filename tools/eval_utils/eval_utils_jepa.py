# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time

import numpy as np
import torch
import tqdm

from mtr.utils import common_utils
from tools.train_utils.train_utils_jepa import get_jepa_loss, get_jepa_loss_with_map


def eval_one_epoch(cfg, context_encoder,
                    predictor,
                    target_encoder,
                    map_predictor, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    context_encoder.eval()
    predictor.eval()
    target_encoder.eval()
    if map_predictor is not None:
        map_predictor.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    sum_loss = 0.0
    sum_mse = 0.0
    sum_std = 0.0
    sum_cov = 0.0
    sum_map_loss = 0.0
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_dict = context_encoder(batch_dict)
            predicted_obj_features = predictor(batch_dict['center_objects_feature'])
            if map_predictor is not None:
                predicted_map_features = map_predictor(batch_dict['map_feature'])
            target_encoding, target_map_encoding = target_encoder(batch_dict, target=True)
            if map_predictor is not None:
                loss, single_losses = get_jepa_loss_with_map(predicted_obj_features, target_encoding, predicted_map_features, target_map_encoding, mse_coeff=cfg.MODEL.CONTEXT_ENCODER.mse_coeff, std_coeff=cfg.MODEL.CONTEXT_ENCODER.std_coeff, cov_coeff=cfg.MODEL.CONTEXT_ENCODER.cov_coeff)
            else:
                loss, single_losses = get_jepa_loss(predicted_obj_features, target_encoding, mse_coeff=cfg.MODEL.CONTEXT_ENCODER.mse_coeff, std_coeff=cfg.MODEL.CONTEXT_ENCODER.std_coeff, cov_coeff=cfg.MODEL.CONTEXT_ENCODER.cov_coeff)
        sum_loss += loss.item()
        sum_mse += single_losses[0].item()
        sum_std += single_losses[1].item()
        sum_cov += single_losses[2].item()
        if map_predictor is not None:
            sum_map_loss += single_losses[3].item()

        disp_dict = {'loss': loss.item()}

        if map_predictor is not None:
            disp_dict.update({'mse_loss': single_losses[0].item(), 'std_loss': single_losses[1].item(), 'cov_loss': single_losses[2].item(), 'map_mse_loss': single_losses[3].item()})
        else:
            disp_dict.update({'mse_loss': single_losses[0].item(), 'std_loss': single_losses[1].item(), 'cov_loss': single_losses[2].item()})

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')
    if cfg.LOCAL_RANK == 0:
        disp_dict.update({'loss': sum_loss / len(dataloader), 'mse_loss': sum_mse / len(dataloader), 'std_loss': sum_std / len(dataloader), 'cov_loss': sum_cov / len(dataloader)})
        if map_predictor is not None:
            disp_dict.update({'map_mse_loss': sum_map_loss / len(dataloader)})

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    ret_dict.update(disp_dict)

    logger.info('****************Evaluation done.*****************')

    return ret_dict


def eval_one_epoch_jepa(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    sum_loss = 0.0
    sum_mse = 0.0
    sum_std = 0.0
    sum_cov = 0.0
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts, eval_loss, sub_losses = model(batch_dict)
            #final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            #pred_dicts += final_pred_dicts

        disp_dict = {}
        sum_loss += eval_loss.item()
        sum_mse += sub_losses[0].item()
        sum_std += sub_losses[1].item()
        sum_cov += sub_losses[2].item()
        disp_dict.update({'eval_loss': sum_loss / (i + 1)})
        disp_dict.update({'mse': sum_mse / (i + 1)})
        disp_dict.update({'std': sum_std / (i + 1)})
        disp_dict.update({'cov': sum_cov / (i + 1)})

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # if dist_test:
    #     logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
    #     pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
    #     logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    ret_dict.update(disp_dict)

    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(pred_dicts, f)

    # result_str, result_dict = dataset.evaluation(
    #     pred_dicts,
    #     output_path=final_output_dir, 
    # )

    # logger.info(result_str)
    # ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
