import os
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
    kde_lidar = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(point_cloud)
    #print("density lidar done")

    # Fit KDE for skeleton keypoints
    kde_skeleton = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(keypoint)
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


def parse_pose_data_kl(scenario, path_to_womd='/home/erik/raid/datasets/womd', path_to_lidar_snippets='/home/erik/raid/datasets/womd/lidar_snippets', path_to_poses='/home/erik/raid/datasets/womd/poses'):
    path_to_poses_kl = '/home/erik/raid/datasets/womd/poses_kl'
    with open(scenario, 'rb') as f:
        scenario_data = pickle.load(f)

    scenario_id = scenario_data['scenario_id']
    if not os.path.exists(os.path.join(path_to_lidar_snippets, str(scenario_id))):
        os.makedirs(os.path.join(path_to_lidar_snippets, str(scenario_id)))
    if not os.path.exists(os.path.join(path_to_poses_kl, str(scenario_id))):
        os.makedirs(os.path.join(path_to_poses_kl, str(scenario_id)))

    # Get agents of interest
    tracks_to_predict = scenario_data['tracks_to_predict']
    object_types = tracks_to_predict['object_type']
    object_indices = tracks_to_predict['track_index']
    indices_of_interest = list()
    for i, object_type in enumerate(object_types):
        if object_type == 'TYPE_CYCLIST' or object_type == 'TYPE_PEDESTRIAN':
            indices_of_interest.append(object_indices[i])
    # Load computed poses
    save_path = os.path.join(path_to_poses, str(scenario_id), 'pose_data.npy')
    pose_data = np.load(save_path)
    object_ids, object_types, trajs = scenario_data['track_infos']['object_id'], scenario_data['track_infos']['object_type'], scenario_data['track_infos']['trajs']
    # TODO: Check feature dimensions of output
    for i, (object_id, object_type) in enumerate(zip(object_ids, object_types)):
        if object_id in indices_of_interest:
            for j in range(11):
                curr_pose = pose_data[i, j, :]
                if np.all(curr_pose == 0):
                    continue
                else:
                    curr_position = trajs[i, j, :3]
                    lidar_file_path = os.path.join(path_to_lidar_snippets, str(scenario_id), str(object_id) + "_" + str(j) + ".npy")
                    if os.path.exists(lidar_file_path):
                        lidar_data = np.load(lidar_file_path)
                        lidar_data = lidar_data - curr_position
                        kl_divergence = calc_kl_divergence(lidar_data, curr_pose)
                        if kl_divergence > 0.1:
                            pose_data[i, j, :, :] = 0
        else:
            continue
    path_to_poses_kl = '/home/erik/raid/datasets/womd/poses_kl'
    save_path = os.path.join(path_to_poses_kl, str(scenario_id), 'pose_data.npy')
    np.save(save_path, pose_data)
    return True


if __name__ == '__main__':
    path_to_womd = '/home/erik/raid/datasets/womd'
    path_to_lidar_snippets = '/home/erik/raid/datasets/womd/lidar_snippets'
    path_to_poses = '/home/erik/raid/datasets/womd/poses'

    scenario_list_training = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_training', '*.pkl'))
    scenario_list_validation = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_validation', '*.pkl'))

    num_workers = 16
    for scenarios in [scenario_list_training, scenario_list_validation]:
        func = partial(parse_pose_data_kl, path_to_womd=path_to_womd, path_to_lidar_snippets=path_to_lidar_snippets, path_to_poses=path_to_poses)

        with multiprocessing.Pool(num_workers) as pool:
            data_infos = list(tqdm(pool.imap(func, scenarios), total=len(scenarios)))
