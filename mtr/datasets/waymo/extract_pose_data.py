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
import torch
import random
sys.path.append('/home/erik/ScePT/ScePT/poses/Pose_Estimation_main/models')
sys.path.append('/home/erik/ScePT/ScePT/poses/Pose_Estimation_main')
from supervised.point_networks.pointnet import PointNet
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)



def calc_kl_divergence(point_cloud, keypoint):
    kde_lidar = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(point_cloud.squeeze().permute(1,0).numpy())
    #print("density lidar done")

    # Fit KDE for skeleton keypoints
    kde_skeleton = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(keypoint.numpy())
    #print("density skeleton done")

    # Evaluate the PDFs on a grid
    grid_points = np.linspace(-3, 3, 100)
    grid = np.array(np.meshgrid(grid_points, grid_points, grid_points)).T.reshape(-1, 3)

    lidar_pdf = np.exp(kde_lidar.score_samples(grid))
    skeleton_pdf = np.exp(kde_skeleton.score_samples(grid))

    #print("scoring done")

    # Normalize the PDFs to form valid probability distributions
    lidar_pdf /= np.sum(lidar_pdf)
    skeleton_pdf /= np.sum(skeleton_pdf)

    # Calculate KL divergence
    kl_divergence = entropy(skeleton_pdf, lidar_pdf)
    return kl_divergence


if __name__ == '__main__':
    path_to_womd = '/home/erik/raid/datasets/womd'
    path_to_lidar_snippets = '/home/erik/raid/datasets/womd/lidar_snippets'
    path_to_poses = '/home/erik/raid/datasets/womd/poses_kl'
    
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
            if not os.path.exists(os.path.join(path_to_lidar_snippets, str(scenario_id))):
                os.makedirs(os.path.join(path_to_lidar_snippets, str(scenario_id)))
            if not os.path.exists(os.path.join(path_to_poses, str(scenario_id))):
                os.makedirs(os.path.join(path_to_poses, str(scenario_id)))
            object_ids, object_types, trajs = scenario_data['track_infos']['object_id'], scenario_data['track_infos']['object_type'], scenario_data['track_infos']['trajs']
            #curr_time_index = scenario_data['current_time_index']
            pose_data_scenario = torch.zeros((len(object_ids), 11, model.num_joints, 3), dtype=torch.float32)
            pose_data_mask = np.zeros((len(object_ids), 11))
            # TODO: Check feature dimensions of output
            pose_features_scenario = torch.zeros((len(object_ids), 11, 64, 64), dtype=torch.float32)
            for i, (object_id, object_type) in enumerate(zip(object_ids, object_types)):
                pose_data = torch.zeros((11, model.num_joints, 3), dtype=torch.float32)
                # TODO: Check feature dimensions of output
                pose_features = torch.zeros((11, 64, 64), dtype=torch.float32)
                if object_type == 'TYPE_CYCLIST' or object_type == 'TYPE_PEDESTRIAN':
                    for j in range(11):
                        curr_position = trajs[i, j, :3]
                        lidar_file_path = os.path.join(path_to_lidar_snippets, str(scenario_id), str(object_id) + "_" + str(j) + ".npy")
                        if os.path.exists(lidar_file_path):
                            lidar_data = np.load(lidar_file_path)
                            lidar_data = lidar_data - curr_position
                            min_num_points = 75
                            if lidar_data.shape[0] >= min_num_points:
                                sample_to = 128
                                if lidar_data.shape[0] > sample_to:
                                    # downsample
                                    indices = random.sample(range(lidar_data.shape[0]), sample_to)
                                    lidar_data = lidar_data[indices]
                                elif lidar_data.shape[0] < sample_to:
                                    # upsample
                                    diff = sample_to-lidar_data.shape[0]
                                    indices = random.choices(range(lidar_data.shape[0]), k=diff)
                                    double_points = lidar_data[indices]
                                    lidar_data = np.concatenate([lidar_data, double_points])
                                point_cloud = torch.from_numpy(lidar_data).unsqueeze(0).permute(0, 2, 1).type(torch.float32).to('cuda')
                                pose, features = model(point_cloud)
                                # Check for Mahalanobis/KL Divergence Distance
                                kl_divergence = calc_kl_divergence(point_cloud.detach().cpu(), pose.detach().cpu()[0, :, :])
                                if kl_divergence < 0.1:
                                    pose_data[j] = pose.detach().cpu()[0, :, :]
                                    pose_features[j] = features.detach().cpu()[0, :, :]
                                    pose_data_mask[i, j] = 1
                scenario_dict[object_id] = pose_data
                pose_data_scenario[i] = pose_data
                pose_features_scenario[i] = pose_features
            save_path = os.path.join(path_to_poses, str(scenario_id), 'pose_data.npy')
            np.save(save_path, pose_data_scenario.numpy())
            save_path = os.path.join(path_to_poses, str(scenario_id), 'pose_data_mask.npy')
            np.save(save_path, pose_data_mask)
            save_path = os.path.join(path_to_poses, str(scenario_id), 'pose_features.npy')
            np.save(save_path, pose_features_scenario.numpy())
