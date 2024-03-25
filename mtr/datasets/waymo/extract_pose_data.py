import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2, compressed_lidar_pb2
from waymo_open_dataset.utils import womd_lidar_utils
from waymo_open_dataset import dataset_pb2
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import tfrecord
import torch
sys.path.append('/home/erik/ScePT/ScePT/poses/Pose_Estimation_main/models')
sys.path.append('/home/erik/ScePT/ScePT/poses/Pose_Estimation_main')
from supervised.point_networks.pointnet import PointNet

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

if __name__ == '__main__':
    path_to_womd = '/home/erik/raid/datasets/womd'
    path_to_lidar_snippets = '/home/erik/raid/datasets/womd/lidar_snippets'
    
    model = PointNet(dropout=0.4) # Model expects input to be batch_size, dim_features, num_points
    # Output is of shape batch_size, num_joints, 3
    model.load_state_dict(torch.load('/home/erik/ScePT/ScePT/poses/runs/run_2024-03-21T10-11-40-101296_lidar_only/ckpts/best_model'))
    model.to('cuda')
    model.eval()

    scenario_list_training = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_training', '*.pkl'))
    scenario_list_validation = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_validation', '*.pkl'))

    for scenario_list in [scenario_list_training, scenario_list_validation]:
        for scenario in tqdm(scenario_list):
            scenario_dict = {}
            with open(scenario, 'rb') as f:
                scenario_data = pickle.load(f)

            scenario_id = scenario_data['scenario_id']
            object_ids, object_types, trajs = scenario_data['track_infos']['object_id'], scenario_data['track_infos']['object_type'], scenario_data['track_infos']['trajs']
            pose_data_scenario = torch.zeros(len(object_ids), 11, model.num_joints, 3)
            pose_data_mask = np.zeros(len(object_ids), 11)
            for i, object_id, object_type in enumerate(zip(object_ids, object_types)):
                pose_data = torch.zeros(11, model.num_joints, 3)
                if object_type == 'TYPE_CYCLIST' or object_type == 'TYPE_PEDESTRIAN':
                    for j in range(11):
                        lidar_file_path = os.path.join(path_to_lidar_snippets, scenario_id, object_id + "_" + str(j) + ".npy")
                        if os.path.exists(lidar_file_path):
                            lidar_data = np.load(lidar_file_path)
                            lidar_data = torch.from_numpy(lidar_data).unsqueeze(0).permute(0, 2, 1).to('cuda')
                            pose, _ = model(lidar_data)
                            pose_data[j] = pose.detach().cpu()[0, :, :]
                            pose_data_mask[i, j] = 1
                scenario_dict[object_id] = pose_data
                pose_data_scenario[i] = pose_data
            save_path = os.path.join(path_to_lidar_snippets, scenario_id, 'pose_data.npy')
            np.save(save_path, pose_data_scenario.numpy())
            save_path = os.path.join(path_to_lidar_snippets, scenario_id, 'pose_data_mask.npy')
            np.save(save_path, pose_data_mask)
