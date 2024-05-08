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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name):
  for laser_calib in frame_lasers.laser_calibrations:
    if laser_calib.name == laser_name:
      return laser_calib
  return None


def get_pc_snippet(points, corners):
    """
    Use points of shape [num_points, 3] and coordinates of the 3D bbox to get the corresponding point cloud snippet and return it.
    points: point cloud
    corners: [8,3] array of 3D coordinates
    """
    # Create a mask for the bounding box using the corners of the 3D bbox
    points = points.astype(np.float32)
    corners = corners.astype(np.float32)
    min_bound = np.min(corners, axis=0)
    max_bound = np.max(corners, axis=0)
    #print(min_bound, max_bound)
    mask = np.all(np.concatenate((np.logical_and(points[:, 0] >= min_bound[0], points[:, 0] <= max_bound[0])[:, np.newaxis], np.logical_and(points[:, 1] >= min_bound[1], points[:, 1] <= max_bound[1])[:, np.newaxis], np.logical_and(points[:, 2] >= min_bound[2], points[:, 2] <= max_bound[2])[:, np.newaxis]), axis=1), axis=1)

    return points[mask, :]


def rotate_points(points, angle, center):
    points = points - center

    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    rotated_points = np.dot(points, rot_matrix.T)
    rotated_points = rotated_points + center

    return rotated_points
def generate_3d_bbox_corners(center, l_w_h, orientation):
    l, w, h = l_w_h
    x, y, z = center

    # Generate the 8 corners of the bounding box
    corners = np.array([
        [x - l/2, y - w/2, z - h/2],
        [x + l/2, y - w/2, z - h/2],
        [x - l/2, y + w/2, z - h/2],
        [x + l/2, y + w/2, z - h/2],
        [x - l/2, y - w/2, z + h/2],
        [x + l/2, y - w/2, z + h/2],
        [x - l/2, y + w/2, z + h/2],
        [x + l/2, y + w/2, z + h/2]
    ])

    corners = rotate_points(corners, orientation, center)
    return corners


def parse_lidar_data(scenario, lidar_path, lidar_save_path):
  with open(scenario, 'rb') as f:
      scenario_dict = pickle.load(f)
  scenario_id = scenario_dict['scenario_id']
  lidar_file = os.path.join(lidar_path, scenario_id + '.tfrecord')
  if os.path.exists(lidar_file):
    dataset = tf.data.TFRecordDataset(lidar_file, compression_type='')
    data = next(iter(dataset))
    lidar_scenario = scenario_pb2.Scenario.FromString(data.numpy())
    frame_points_xyz = dict()
    frame_points_feature = dict()
    frame_i = 0

    for frame_lasers in lidar_scenario.compressed_frame_laser_data:
      points_xyz_list = []
      points_feature_list = []
      # frame_pose = np.reshape(np.array(
      #     lidar_scenario.compressed_frame_laser_data[frame_i].pose.transform),
      #     (4, 4))
      frame_pose = np.eye(4)
      for laser in frame_lasers.lasers:
        if laser.name == dataset_pb2.LaserName.TOP:
          c = _get_laser_calib(frame_lasers, laser.name)
          (points_xyz, points_feature,
          points_xyz_return2,
          points_feature_return2) = womd_lidar_utils.extract_top_lidar_points(
              laser, frame_pose, c)
        else:
          c = _get_laser_calib(frame_lasers, laser.name)
          (points_xyz, points_feature,
          points_xyz_return2,
          points_feature_return2) = womd_lidar_utils.extract_side_lidar_points(
              laser, c)
        points_xyz_list.append(points_xyz.numpy())
        points_xyz_list.append(points_xyz_return2.numpy())
        points_feature_list.append(points_feature.numpy())
        points_feature_list.append(points_feature_return2.numpy())
      frame_points_xyz[frame_i] = np.concatenate(points_xyz_list, axis=0)
      frame_points_feature[frame_i] = np.concatenate(points_feature_list, axis=0)
      frame_i += 1
    
    track_infos = scenario_dict['track_infos']
    for i in range(track_infos['trajs'].shape[0]):
      # Lidar is available for 11 timesteps which is the first second and the current timestep
      obj_id = track_infos['object_id'][i]
      for j in range(11):
        valid = track_infos['trajs'][i, j, -1]
        if valid:
          center = track_infos['trajs'][i, j, :3]
          l_w_h = track_infos['trajs'][i, j, 3:6]
          orientation = track_infos['trajs'][i, j, 6]
          snippet = get_pc_snippet(frame_points_xyz[j], generate_3d_bbox_corners(center, l_w_h, orientation))
          lidar_scenario_folder = os.path.join(lidar_save_path, scenario_id)
          if not os.path.exists(lidar_scenario_folder):
            os.makedirs(lidar_scenario_folder)
          lidar_snippet_path = os.path.join(lidar_scenario_folder, str(obj_id)+'_'+str(j)+'.npy')
          np.save(lidar_snippet_path, snippet)
  else:
     print(f'Missing lidar file for scenario {scenario_id}')
  return True

if __name__ == '__main__':
    path_to_womd = '/home/erik/raid/datasets/womd'
    #lidar_path_training = os.path.join(path_to_womd, 'v_1_2_1/uncompressed/lidar_and_camera/training')
    lidar_path_validation = os.path.join('/home/erik/NAS/publicdatasets/waymo_motion', 'v_1_2_1/uncompressed/lidar_and_camera/validation')
    #scenario_list_training = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_training', '*.pkl'))
    scenario_list_validation = glob.glob(os.path.join(path_to_womd, 'processed_scenarios_validation_interactive', '*.pkl'))
    lidar_save_path = os.path.join(path_to_womd, 'lidar_snippets')

    if not os.path.exists(lidar_save_path):
        os.makedirs(lidar_save_path)

    num_workers = 16
    for lidar, scenarios in zip([lidar_path_validation], [scenario_list_validation]):
      func = partial(parse_lidar_data, lidar_save_path=lidar_save_path, lidar_path=lidar)

      with multiprocessing.Pool(num_workers) as p:
          data_infos = list(tqdm(p.imap(func, scenarios), total=len(scenarios)))
    #parse_lidar_data(scenario_list_training, lidar_path_training, lidar_save_path)