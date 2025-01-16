# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.utils import common_utils
from mtr.models import model as model_utils
from mtr.models.context_encoder import build_context_encoder
import copy

from train_utils.train_utils_jepa import train_model
from train_utils.train_utils_jepa import save_checkpoint, load_checkpoint


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=None, required=False, help='batch size for evaluation if different')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    parser.add_argument('--single_overfit', type=int, default=0, help='Number of samples of training set used for overfitting')
    parser.add_argument('--show_grad_curve', action='store_true', default=False, help='Show grad curve')
    parser.add_argument('--scenario_id', nargs='+', default=None, help='scenario ids for subset of dataset')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def build_optimizer(model, opt_cfg):
    if opt_cfg.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
        )
    elif opt_cfg.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
    else:
        assert False

    return optimizer


def build_scheduler(optimizer, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch, it=0):
    decay_steps = [x * total_iters_each_epoch for x in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)
    
    if opt_cfg.get('SCHEDULER', None) == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2 * total_iters_each_epoch,
            T_mult=1,
            eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
            last_epoch=-1,
        )
    elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
        scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = lr_sched.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, total_iters=total_iters, last_epoch=last_epoch)
    elif opt_cfg.get('SCHEDULER', None) == 'jepa_cosine':
        scheduler = common_utils.WarmupCosineSchedule(optimizer,
                                                      warmup_steps=opt_cfg.get('WARMUP_EPOCHS', 0) * total_iters_each_epoch,
                                                      start_lr=opt_cfg.get('LR', 0.0001),
                                                      ref_lr=opt_cfg.get('REF_LR', 0.001),
                                                      final_lr=opt_cfg.get('FINAL_LR', 0.000001),
                                                      T_max=total_epochs * total_iters_each_epoch,
                                                    )
    else:
        scheduler = None

    return scheduler


def build_scaler():
    return torch.cuda.amp.GradScaler()


def init_opt(
        logger,
        config,
        encoder,
        predictor,
        iterations_per_epoch,
        num_epochs,
        map_predictor=None,
        ipe_scale=1.25,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    if map_predictor is not None:
        param_groups.append(
            {
                'params': (p for n, p in map_predictor.named_parameters()
                            if ('bias' not in n) and (len(p.shape) != 1))
            }
        )
        param_groups.append(
            {
                'params': (p for n, p in map_predictor.named_parameters()
                            if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        )
    logger.info('Generating optimizer...')
    optimizer = torch.optim.AdamW(param_groups, lr=config.LR, weight_decay=config.get('WEIGHT_DECAY', 0))
    logger.info('Generating scheduler...')
    scheduler = build_scheduler(optimizer, config, num_epochs, iterations_per_epoch, -1, 0)
    logger.info('Generating weight decay scheduler...')
    wd_scheduler = common_utils.CosineWDSchedule(
        optimizer,
        ref_wd=config.WEIGHT_DECAY,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        final_wd=config.get('FINAL_WEIGHT_DECAY', config.WEIGHT_DECAY)
    )
    if config.get('use_scaler', False):
        scaler = build_scaler()
    else:
        scaler = None

    return optimizer, scaler, scheduler, wd_scheduler


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
        args.without_sync_bn = True
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    #logger.info('slurmprocid: %d' % int(os.environ['SLURM_PROCID']))

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    ################################### Build Dataloader ###################################

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
        single_overfit=args.single_overfit,
        scenario_id=args.scenario_id
    )

    ipe = len(train_loader)

    ################################### Build Model ###################################

    context_encoder = build_context_encoder(cfg.MODEL.CONTEXT_ENCODER)
    predictor = model_utils.build_jepa_predictor(cfg.MODEL.CONTEXT_ENCODER)
    for m in context_encoder.modules():
        model_utils.init_weights(m, std=0.02)
    for m in predictor.modules():
        model_utils.init_weights(m, std=0.02)
    if cfg.MODEL.CONTEXT_ENCODER.get('USE_MAP_LOSS', False):
        map_predictor = model_utils.build_map_predictor(cfg.MODEL.CONTEXT_ENCODER)
        for m in map_predictor.modules():
            model_utils.init_weights(m, std=0.02)
    else:
        map_predictor = None
    # JEPA puts the models onto devices here. If we want to go full SLURM we might also have to do that

    if not args.without_sync_bn:
        context_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(context_encoder)
        predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
        if cfg.MODEL.CONTEXT_ENCODER.get('USE_MAP_LOSS', False):
            map_predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(map_predictor)

    context_encoder.cuda()
    predictor.cuda()
    if map_predictor is not None:
        map_predictor.cuda()

    target_encoder = copy.deepcopy(context_encoder)
    # for p in target_encoder.parameters():
    #     p.requires_grad = False

    if cfg.LOCAL_RANK == 0:

        total_params = sum(p.numel() for p in context_encoder.parameters())
        print(f"Total number of parameters in context encoder: {total_params}")

        total_params = sum(p.numel() for p in predictor.parameters())
        print(f"Total number of parameters in predictor: {total_params}")

        if map_predictor is not None:
            total_params = sum(p.numel() for p in map_predictor.parameters())
            print(f"Total number of parameters in map predictor: {total_params}")

    ################################### Build Optimizer ###################################

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        logger,
        cfg.OPTIMIZATION,
        context_encoder,
        predictor,
        len(train_loader),
        args.epochs,
        map_predictor=map_predictor,
        ipe_scale=cfg.OPTIMIZATION.get('IPE_SCALE', 1.25)
    )

    momentum_scheduler = (cfg.OPTIMIZATION.ema[0] + i * (cfg.OPTIMIZATION.ema[1] - cfg.OPTIMIZATION.ema[0]) / (ipe * args.epochs * cfg.OPTIMIZATION.ipe_scale) for i in range(int(ipe*args.epochs*cfg.OPTIMIZATION.ipe_scale)+1))

    ################################### Load Checkpoint ###################################

    # load checkpoint if it is possible
    start_epoch = it = 0

    ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        while len(ckpt_list) > 0:
            basename = os.path.basename(ckpt_list[-1])
            if basename == 'best_model.pth':
                ckpt_list = ckpt_list[:-1]
                continue

            try:
                context_encoder, predictor, target_encoder, map_predictor, optimizer, scaler, start_epoch = load_checkpoint(
                    ckpt_list[-1],
                    optimizer,
                    scaler,
                    context_encoder,
                    predictor,
                    target_encoder,
                    logger,
                    map_predictor
                )
                for _ in range(start_epoch * ipe):
                    scheduler.step()
                    wd_scheduler.step()
                    next(momentum_scheduler)

                break
            except:
                ckpt_list = ckpt_list[:-1]

    ################################### Wrap DDP ###################################
    
    if dist_train:
        context_encoder = nn.parallel.DistributedDataParallel(context_encoder, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], static_graph=True)
        context_encoder.train()
        predictor = nn.parallel.DistributedDataParallel(predictor, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], static_graph=True)
        predictor.train()
        if map_predictor is not None:
            map_predictor = nn.parallel.DistributedDataParallel(map_predictor, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], static_graph=True)
            map_predictor.train()
        target_encoder = nn.parallel.DistributedDataParallel(target_encoder, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        for p in target_encoder.parameters():
            p.requires_grad = False
        target_encoder.eval()
    if cfg.LOCAL_RANK == 0:
        logger.info('Context Encoder: ')
        logger.info(context_encoder)
        logger.info('Predictor: ')
        logger.info(predictor)
        logger.info('Target Encoder: ')
        logger.info(target_encoder)
        if map_predictor is not None:
            logger.info('Map Predictor: ')
            logger.info(map_predictor)

    
    ################################### Eval Loader ###################################

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size if args.single_overfit == 0 else args.eval_batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    ################################### Start Training Loop ###################################
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
 
    train_model(
        context_encoder,
        predictor,
        target_encoder,
        map_predictor,
        optimizer,
        scaler,
        scheduler,
        wd_scheduler,
        train_loader,
        optim_cfg=cfg.OPTIMIZATION,
        momentum_scheduler=momentum_scheduler,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=start_epoch*ipe,
        rank=cfg.LOCAL_RANK,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        tb_log=tb_log,
        logger=logger,
        eval_output_dir=eval_output_dir,
        test_loader=test_loader if not args.not_eval_with_train else None,
        cfg=cfg,
        dist_train=dist_train,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        show_grad_curve=args.show_grad_curve,
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()
