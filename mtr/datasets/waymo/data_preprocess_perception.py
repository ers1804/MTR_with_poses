# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type

# Imports for new dataset structure (Waymo V2)
import dask.dataframe as dd
from waymo_open_dataset import v2

tf.compat.v1.enable_eager_execution()

    
def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'trajs': []
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)

    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[cur_data.lane.type]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': road_line_type[x.boundary_type]  # roadline type
                } for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            print(cur_data)
            raise ValueError

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos


def process_waymo_data_with_waymo_frame(data_file, output_path=None):
    # Save variables
    ret_infos = []
    info = {}
    # Use v2 of the dataset
    base_name = os.path.basename(data_file)
    scenario_id = os.path.splitext(base_name)[0]
    # Read parquet files
    lidar_box_df = dd.read_parquet(data_file)
    cam_box_df = dd.read_parquet(data_file.replace('lidar_box', 'camera_box'))
    association_df = dd.read_parquet(data_file.replace('lidar_box', 'camera_to_lidar_box_association'))
    cam_img_df = dd.read_parquet(data_file.replace('lidar_box', 'camera_image'))
    lidar_df = dd.read_parquet(data_file.replace('lidar_box', 'lidar'))

    # Merge the dataframes such that we have all available objects per frame (one frame is one row)
    cam_box_w_association_df = v2.merge(cam_box_df, association_df, left_nullable=True, right_nullable=True)
    cam_box_w_association_lidar_box_df = v2.merge(cam_box_w_association_df, lidar_box_df, left_nullable=True, right_nullable=True)
    full_w_cam_img_df = v2.merge(cam_box_w_association_lidar_box_df, cam_img_df, right_nullable=True)
    full_df = v2.merge(full_w_cam_img_df, lidar_df, right_nullable=True)

    #lidar_box_df = (lidar_box_df.groupby(['key.segment_context_name', 'key.laser_object_id']).agg(list).reset_index())
    # Iterate over objects
    unique_laser_object_ids = list()
    unique_camera_object_ids = list()
    object_infos = dict()
    object_infos['lidar'] = dict()
    object_infos['camera'] = dict()
    for _, row in full_df.iterrows():
        lidar_box = v2.LiDARBoxComponent.from_dict(row)
        cam_box = v2.CameraBoxComponent.from_dict(row)
        cam_img = v2.CameraImageComponent.from_dict(row)
        lidar = v2.LiDARComponent.from_dict(row)
        association = v2.CameraToLiDARBoxAssociationComponent.from_dict(row)
        for lidar_obj_id, cam_obj_id, timestamp, lidar_center, lidar_size, lidar_speed, lidar_heading, lidar_obj_type, cam_center, cam_size, cam_obj_type in zip(association.key.laser_object_id, association.key.camera_object_id, lidar_box.key.frame_timestamp_micros, lidar_box.box.center, lidar_box.box.size, lidar_box.box.speed, lidar_box.box.heading, lidar_box.type, cam_box.box.center, cam_box.box.size, cam_box.type):
            if lidar_obj_id not in unique_laser_object_ids and lidar_obj_id is not None:
                unique_laser_object_ids.append(lidar_obj_id)
                if lidar_obj_id is not None and cam_obj_id is not None:
                    object_infos['lidar'][lidar_obj_id] = dict()
                    object_infos['lidar'][lidar_obj_id]['timestamps'] = list()
                    object_infos['lidar'][lidar_obj_id]['positions'] = list()
                    object_infos['lidar'][lidar_obj_id]['sizes'] = list()
                    object_infos['lidar'][lidar_obj_id]['speeds'] = list()
                    object_infos['lidar'][lidar_obj_id]['headings'] = list()
                    object_infos['lidar'][lidar_obj_id]['object_type'] = lidar_obj_type
                    object_infos['lidar'][lidar_obj_id]['camera_object_id'] = cam_obj_id
                    object_infos['lidar'][lidar_obj_id]['camera_centers'] = list()
                    object_infos['lidar'][lidar_obj_id]['camera_sizes'] = list()
            if cam_obj_id not in unique_camera_object_ids and cam_obj_id is not None:
                unique_camera_object_ids.append(cam_obj_id)
                if lidar_obj_id is None and cam_obj_id is not None:
                    object_infos['camera'][cam_obj_id] = dict()
                    object_infos['camera'][cam_obj_id]['timestamps'] = list()
                    object_infos['camera'][cam_obj_id]['positions'] = list()
                    object_infos['camera'][cam_obj_id]['sizes'] = list()
                    object_infos['camera'][cam_obj_id]['object_type'] = cam_obj_type
            if lidar_obj_id is not None and cam_obj_id is not None:
                # Both lidar and camera object are available
                object_infos['lidar'][lidar_obj_id]['time_stamps'].append(timestamp)
                object_infos['lidar'][lidar_obj_id]['positions'].append(lidar_center)
                object_infos['lidar'][lidar_obj_id]['sizes'].append(lidar_size)
                object_infos['lidar'][lidar_obj_id]['speeds'].append(lidar_speed)
                object_infos['lidar'][lidar_obj_id]['headings'].append(lidar_heading)
                object_infos['lidar'][lidar_obj_id]['camera_centers'].append(cam_center)
                object_infos['lidar'][lidar_obj_id]['camera_sizes'].append(cam_size)

    # But map data has to be loaded from v1 format
    for record_file in data_file:
        dataset = tf.data.TFRecordDataset(record_file, compression_type='')
        ret_infos = []
        info = {}
        timestamps_micros = []

        for cnt, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            #info['context_name'] = frame.context.name # WOMD: Scenario-ID

            #info['timestamps_micros'] = frame.timestamp_micros  # single timestamp since only one frame
            timestamps_micros.append(frame.timestamp_micros)
            #info['current_time_index'] = frame.timestamp_micros  # int, 10

            #info['sdc_track_index'] = scenario.sdc_track_index  # int
            
            info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list

            info['tracks_to_predict'] = {
                'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
                'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
            }  # for training: suggestion of objects to train on, for val/test: need to be predicted

            track_infos = decode_tracks_from_proto(scenario.tracks)
            info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']]

            # decode map related data
            map_infos = decode_map_features_from_proto(scenario.map_features)
            dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

            save_infos = {
                'track_infos': track_infos,
                'dynamic_map_infos': dynamic_map_infos,
                'map_infos': map_infos
            }
            save_infos.update(info)

            output_file = os.path.join(output_path, f'sample_{scenario.scenario_id}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(save_infos, f)

            ret_infos.append(info)
        return ret_infos


def get_infos_from_protos(data_path, output_path=None, num_workers=8):
    from functools import partial
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    func = partial(
        process_waymo_data_with_waymo_frame, output_path=output_path
    )

    sensor_component = 'lidar_box'
    sensor_path = os.path.join(data_path, sensor_component)
    src_files = glob.glob(os.path.join(sensor_path, '*.parquet'))
    src_files.sort()

    # func(src_files[0])
    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))

    all_infos = [item for infos in data_infos for item in infos]
    return all_infos


def create_infos_from_protos(raw_data_path, output_path, num_workers=1):
    train_infos = get_infos_from_protos(
        data_path=os.path.join(raw_data_path, 'training'),
        output_path=os.path.join(output_path, 'processed_scenarios_training'),
        num_workers=num_workers
    )
    train_filename = os.path.join(output_path, 'processed_scenarios_training_infos.pkl')
    with open(train_filename, 'wb') as f:
        pickle.dump(train_infos, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    val_infos = get_infos_from_protos(
        data_path=os.path.join(raw_data_path, 'validation'),
        output_path=os.path.join(output_path, 'processed_scenarios_validation'),
        num_workers=num_workers
    )
    val_filename = os.path.join(output_path, 'processed_scenarios_val_infos.pkl')
    with open(val_filename, 'wb') as f:
        pickle.dump(val_infos, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)
    

if __name__ == '__main__':
    create_infos_from_protos(
        raw_data_path=sys.argv[1],
        output_path=sys.argv[2]
    )
