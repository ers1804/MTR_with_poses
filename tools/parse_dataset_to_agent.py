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

from train_utils.train_utils import train_model
import pickle
import numpy as np
import copy
import tqdm
import multiprocessing
from functools import partial


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--num_workers', type=int, default=8, help='number of processes for parsing')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def filter_info_by_object_type(infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        return ret_infos


def get_all_infos(config, info_path, mode):
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::config.DATA_CONFIG.SAMPLE_INTERVAL[mode]]

        # If scenario id is provided only use the infos of the single scenario

        for func_name, val in config.DATA_CONFIG.INFO_FILTER_DICT.items():
            infos = globals()[func_name](infos, val)

        return infos


def parse_info(info, mode, data_path, output_path, cfg):
    scene_id = info['scenario_id']
    with open(data_path / f'sample_{scene_id}.pkl', 'rb') as f:
        info_pickle = pickle.load(f)
    output = []
    for i in range(len(info_pickle['tracks_to_predict']['track_index'])):
        new_info = copy.deepcopy(info_pickle)
        for key, value in info_pickle['tracks_to_predict'].items():
            new_info['tracks_to_predict'][key] = [value[i]]
        filename = os.path.join(output_path, cfg.DATA_CONFIG.SPLIT_DIR[mode], f'sample_{scene_id}_{i}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(new_info, f)
        new_src_info = copy.deepcopy(info)
        new_src_info['scenario_id'] = f'{scene_id}_{i}'
        for key, value in info['tracks_to_predict'].items():
            new_src_info['tracks_to_predict'][key] = [value[i]]
        output.append(new_src_info)
    return output

def main():
    args, cfg = parse_config()
    data_root = cfg.ROOT_DIR / cfg.DATA_CONFIG.DATA_ROOT

    for mode in ['train', 'test']:
        data_path = data_root / cfg.DATA_CONFIG.SPLIT_DIR[mode]
        infos = get_all_infos(cfg, data_root / cfg.DATA_CONFIG.INFO_FILE[mode], mode)
        new_src_infos = []
        func = partial(
            parse_info, mode=mode, data_path=data_path, output_path=args.output_path, cfg=cfg
        )
        with multiprocessing.Pool(args.num_workers) as p:
            new_src_infos = [
                item for infos in tqdm.tqdm(p.imap(func, infos), total=len(infos))
                for item in infos
            ]
        filename = os.path.join(args.output_path, f'processed_scenarios_{mode}_infos_agent.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(new_src_infos, f)
    

if __name__ == '__main__':
    main()
