# Dataset class for pedestrian pose prediction from pre-parsed Waymo Open Dataset
# Expects per-pedestrian .npz files with SMPL parameters organized by scene


import numpy as np
from pathlib import Path
import torch

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg


# Waymo dataset timestamps: 91 steps at 10Hz covering [0, 9.0] seconds
# current_time_index = 10 (the 11th step, at t=1.0s)
# past: indices 0..10 (11 steps), future: indices 11..90 (80 steps)
WAYMO_TOTAL_TIMESTAMPS = 91
WAYMO_CURRENT_TIME_INDEX = 10
WAYMO_NUM_PAST = WAYMO_CURRENT_TIME_INDEX + 1  # 11
WAYMO_NUM_FUTURE = WAYMO_TOTAL_TIMESTAMPS - WAYMO_NUM_PAST  # 80
WAYMO_DT = 0.1  # 10Hz
WAYMO_TIMESTAMPS = np.arange(WAYMO_TOTAL_TIMESTAMPS) * WAYMO_DT  # [0.0, 0.1, ..., 9.0]

# Pedestrian bounding box defaults (width, length, height in meters)
PEDESTRIAN_SIZE = np.array([0.8, 0.8, 1.7], dtype=np.float32)  # dx, dy, dz


def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: (..., 3) axis-angle vectors.
    Returns:
        rot_mat: (..., 3, 3) rotation matrices.
    """
    batch_shape = axis_angle.shape[:-1]
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # (..., 1)
    angle = np.clip(angle, a_min=1e-8, a_max=None)
    axis = axis_angle / angle  # (..., 3)

    cos_a = np.cos(angle)[..., 0]  # (...)
    sin_a = np.sin(angle)[..., 0]  # (...)

    # skew-symmetric matrix K
    K = np.zeros(batch_shape + (3, 3), dtype=axis_angle.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    eye = np.zeros_like(K)
    eye[..., 0, 0] = 1
    eye[..., 1, 1] = 1
    eye[..., 2, 2] = 1

    rot_mat = eye + sin_a[..., None, None] * K + (1 - cos_a[..., None, None]) * np.einsum('...ij,...jk->...ik', K, K)
    return rot_mat


def rotation_matrix_to_6d(rot_mat):
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        rot_mat: (..., 3, 3)
    Returns:
        repr_6d: (..., 6)
    """
    return np.concatenate([rot_mat[..., :, 0], rot_mat[..., :, 1]], axis=-1)


def smpl_params_to_6d(root_orient, pose_body):
    """Convert SMPL axis-angle parameters to 6D rotation representation.

    Args:
        root_orient: (T, 3) root orientation in axis-angle
        pose_body: (T, 69) body pose in axis-angle (23 joints × 3)
    Returns:
        pose_6d: (T, 144) = 24 joints × 6D (1 root + 23 body)
    """
    T = root_orient.shape[0]
    # SMPL has 24 joints: 1 root + 23 body joints
    num_body_joints = pose_body.shape[-1] // 3  # 69 / 3 = 23

    # Convert root
    root_rot = axis_angle_to_rotation_matrix(root_orient)  # (T, 3, 3)
    root_6d = rotation_matrix_to_6d(root_rot)  # (T, 6)

    # Convert body joints
    body_aa = pose_body.reshape(T, num_body_joints, 3)  # (T, 23, 3)
    body_rot = axis_angle_to_rotation_matrix(body_aa)  # (T, 23, 3, 3)
    body_6d = rotation_matrix_to_6d(body_rot)  # (T, 23, 6)
    body_6d = body_6d.reshape(T, num_body_joints * 6)  # (T, 138)

    pose_6d = np.concatenate([root_6d, body_6d], axis=-1)  # (T, 144)

    return pose_6d.astype(np.float32)


def compute_heading_from_positions(positions):
    """Compute heading angle from sequential positions.

    Args:
        positions: (T, 3) global positions
    Returns:
        heading: (T,) heading angles in radians
    """
    T = positions.shape[0]
    heading = np.zeros(T, dtype=np.float32)
    if T < 2:
        return heading
    # compute heading from displacement
    diff = np.diff(positions[:, :2], axis=0)  # (T-1, 2)
    angles = np.arctan2(diff[:, 1], diff[:, 0])  # (T-1,)
    heading[1:] = angles
    heading[0] = heading[1]
    return heading


def compute_velocity_from_positions(positions, dt=WAYMO_DT):
    """Compute velocity from positions via finite differences.

    Args:
        positions: (T, 3) global positions
        dt: time step
    Returns:
        velocity: (T, 2) velocities (vx, vy)
    """
    T = positions.shape[0]
    velocity = np.zeros((T, 2), dtype=np.float32)
    if T < 2:
        return velocity
    diff = np.diff(positions[:, :2], axis=0)  # (T-1, 2)
    velocity[1:] = diff / dt
    velocity[0] = velocity[1]
    return velocity


class WaymoPoseDataset(DatasetTemplate):
    """Dataset for pedestrian motion + pose prediction from pre-parsed Waymo data.

    Expected data layout:
        <data_root>/<split_dir>/
            <scene_context_id>/
                <pedestrian_id>.npz   # contains trans, root_orient, pose_body, betas, waymo_timestamps

    Each .npz file contains:
        - trans: (N, 3) global translations
        - root_orient: (N, 3) root orientation in axis-angle
        - pose_body: (N, 69) SMPL body pose in axis-angle (23 joints)
        - betas: (10,) SMPL shape parameters
        - waymo_timestamps: (N,) timestamps in seconds
    """

    def __init__(self, dataset_cfg, training=True, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]

        self.num_past = self.dataset_cfg.get('NUM_PAST_TIMESTAMPS', WAYMO_NUM_PAST)  # 11
        self.num_future = self.dataset_cfg.get('NUM_FUTURE_TIMESTAMPS', WAYMO_NUM_FUTURE)  # 80
        self.total_timestamps = self.num_past + self.num_future
        self.current_time_index = self.num_past - 1
        self.dt = self.dataset_cfg.get('DT', WAYMO_DT)

        self.without_hdmap = self.dataset_cfg.get('WITHOUT_HDMAP', True)

        self.infos = self._build_scene_index()
        self.logger.info(f'Total scenes: {len(self.infos)} | Mode: {self.mode}')

    def _build_scene_index(self):
        """Build an index of all scenes, each with all its pedestrian .npz files."""
        scene_dirs = sorted([
            d for d in self.data_path.iterdir() if d.is_dir()
        ])

        sample_interval = self.dataset_cfg.get('SAMPLE_INTERVAL', {}).get(self.mode, 1)

        infos = []
        total_peds = 0
        for scene_dir in scene_dirs:
            npz_files = sorted(list(scene_dir.glob('*.npz')))
            if len(npz_files) == 0:
                continue

            scene_info = {
                'scenario_id': scene_dir.name,
                'scene_dir': str(scene_dir),
                'npz_files': [str(f) for f in npz_files],
                'num_pedestrians': len(npz_files),
            }
            infos.append(scene_info)
            total_peds += len(npz_files)

        infos = infos[::sample_interval]
        self.logger.info(f'Total pedestrians across all scenes: {total_peds}')
        return infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        return self.create_scene_level_data(index)

    def _load_pedestrian(self, npz_path):
        """Load a single pedestrian .npz file and map to the standard Waymo time grid.

        Returns:
            traj_full: (total_timestamps, 10) [cx, cy, cz, dx, dy, dz, heading, vx, vy, valid]
            pose_6d_full: (total_timestamps, 144) 6D rotation representation
            betas: (10,) shape parameters
        """
        data = np.load(npz_path)
        trans = data['trans'].astype(np.float32)           # (N, 3)
        root_orient = data['root_orient'].astype(np.float32)  # (N, 3)
        pose_body = data['pose_body'].astype(np.float32)   # (N, 69)
        betas = data['betas'].astype(np.float32)           # (10,)
        timestamps = data['waymo_timestamps'].astype(np.float64)  # (N,)

        N = len(timestamps)

        # Build the standard time grid
        time_grid = np.arange(self.total_timestamps) * self.dt  # [0.0, 0.1, ..., 9.0]

        # Map each observed timestamp to the nearest grid index
        traj_full = np.zeros((self.total_timestamps, 10), dtype=np.float32)
        pose_6d_full = np.zeros((self.total_timestamps, 144), dtype=np.float32)

        # Convert pose params to 6D
        pose_6d = smpl_params_to_6d(root_orient, pose_body)  # (N, 144)

        # Compute heading and velocity from positions
        heading = compute_heading_from_positions(trans)  # (N,)
        velocity = compute_velocity_from_positions(trans, dt=self.dt)  # (N, 2)

        # For each observed timestep, find closest grid slot
        for i in range(N):
            grid_idx = np.argmin(np.abs(time_grid - timestamps[i]))
            if grid_idx < self.total_timestamps:
                traj_full[grid_idx, 0:3] = trans[i]  # cx, cy, cz
                traj_full[grid_idx, 3:6] = PEDESTRIAN_SIZE  # dx, dy, dz
                traj_full[grid_idx, 6] = heading[i]  # heading
                traj_full[grid_idx, 7:9] = velocity[i]  # vx, vy
                traj_full[grid_idx, 9] = 1.0  # valid

                pose_6d_full[grid_idx] = pose_6d[i]

        return traj_full, pose_6d_full, betas

    def create_scene_level_data(self, index):
        """Create scene-level data matching the WaymoDataset output interface.

        Returns a dict with the same keys as WaymoDataset.__getitem__ plus pose-related keys.
        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        npz_files = info['npz_files']
        num_objects = len(npz_files)

        # Load all pedestrians in the scene
        obj_trajs_full_list = []
        pose_6d_full_list = []
        betas_list = []
        obj_ids = []

        for npz_path in npz_files:
            traj_full, pose_6d_full, betas = self._load_pedestrian(npz_path)
            obj_trajs_full_list.append(traj_full)
            pose_6d_full_list.append(pose_6d_full)
            betas_list.append(betas)
            obj_ids.append(Path(npz_path).stem)

        obj_trajs_full = np.stack(obj_trajs_full_list, axis=0)  # (num_objects, total_timestamps, 10)
        pose_6d_full = np.stack(pose_6d_full_list, axis=0)  # (num_objects, total_timestamps, 144)
        betas_all = np.stack(betas_list, axis=0)  # (num_objects, 10)

        obj_types = np.array(['TYPE_PEDESTRIAN'] * num_objects)
        obj_ids = np.array(obj_ids)

        # Split into past and future
        obj_trajs_past = obj_trajs_full[:, :self.num_past]  # (num_objects, 11, 10)
        obj_trajs_future = obj_trajs_full[:, self.num_past:]  # (num_objects, 80, 10)

        pose_past = pose_6d_full[:, :self.num_past]  # (num_objects, 11, 144)
        pose_future = pose_6d_full[:, self.num_past:]  # (num_objects, 80, 144)

        timestamps = np.arange(self.num_past, dtype=np.float32) * self.dt

        # Determine which pedestrians to predict: those valid at current_time_index
        valid_at_current = obj_trajs_full[:, self.current_time_index, -1] > 0  # (num_objects,)
        track_index_to_predict = np.where(valid_at_current)[0]

        if len(track_index_to_predict) == 0:
            # Fallback: pick the first pedestrian if none are valid at current time
            track_index_to_predict = np.array([0])

        # Get center objects (state at current time)
        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=self.current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )

        # Use the first valid pedestrian as a pseudo-SDC
        sdc_track_index = track_index_to_predict[0]

        # Create agent data (centered trajectories, masks, future states)
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos,
         obj_trajs_future_state, obj_trajs_future_mask,
         center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new, sdc_track_index_new,
         obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict,
            sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        # Create pose data for center objects
        # We need to apply the same valid_past_mask filtering as create_agent_data_for_center_objects
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects,)

        # Filter poses and betas to match filtered objects
        pose_past_filtered = pose_past[valid_past_mask]  # (num_filtered_objects, 11, 144)
        pose_future_filtered = pose_future[valid_past_mask]  # (num_filtered_objects, 80, 144)
        betas_filtered = betas_all[valid_past_mask]  # (num_filtered_objects, 10)

        # Create pose tensors for center objects
        num_center_objects = len(track_index_to_predict_new)

        # obj_poses: (num_center_objects, num_objects, num_past_timestamps, 144)
        # Need to replicate for each center object (same as obj_trajs_data)
        obj_poses = np.tile(pose_past_filtered[None], (num_center_objects, 1, 1, 1))  # (num_center, num_obj, 11, 144)

        # Pose mask: same as trajectory valid mask for past
        obj_trajs_past_filtered = obj_trajs_past[valid_past_mask]  # (num_filtered_objects, 11, 10)
        obj_poses_mask = np.tile(
            (obj_trajs_past_filtered[:, :, -1] > 0)[None],
            (num_center_objects, 1, 1)
        )  # (num_center, num_obj, 11)

        # Center ground truth poses (future poses for the center objects)
        center_gt_poses = pose_future_filtered[track_index_to_predict_new]  # (num_center, 80, 144)

        # Apply the future mask to zero out invalid timesteps
        center_gt_poses_masked = center_gt_poses.copy()
        center_gt_poses_masked[center_gt_trajs_mask == 0] = 0

        # Center shape parameters
        center_shape_params = betas_filtered[track_index_to_predict_new]  # (num_center, 10)

        # Build return dict (matching WaymoDataset interface)
        ret_dict = {
            'scenario_id': np.array([scene_id] * num_center_objects),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': obj_ids[track_index_to_predict_new],
            'center_objects_type': np.array(['TYPE_PEDESTRIAN'] * num_center_objects),

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict],

            # Pose-specific keys
            'obj_poses': obj_poses.astype(np.float32),
            'obj_poses_mask': obj_poses_mask.astype(bool),
            'center_gt_poses': center_gt_poses_masked.astype(np.float32),
            'center_shape_params': center_shape_params.astype(np.float32),
        }

        # Always provide map data — empty placeholders when no HD map is available.
        # The encoder unconditionally reads map_polylines / map_polylines_mask,
        # so these keys must always be present.
        num_polylines = 2  # minimal placeholder
        num_points_each_polyline = self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20)
        ret_dict['map_polylines'] = np.zeros(
            (num_center_objects, num_polylines, num_points_each_polyline, 9), dtype=np.float32
        )
        ret_dict['map_polylines_mask'] = np.zeros(
            (num_center_objects, num_polylines, num_points_each_polyline), dtype=bool
        )
        ret_dict['map_polylines_center'] = np.zeros(
            (num_center_objects, num_polylines, 3), dtype=np.float32
        )

        return ret_dict

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        """Select agents that are valid at the current time index."""
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]
            if obj_trajs_full[obj_idx, current_time_index, -1] > 0:
                center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
                track_index_to_predict_selected.append(obj_idx)

        if len(center_objects_list) == 0:
            # Fallback: use the first object with any valid timestep
            for obj_idx in range(len(obj_trajs_full)):
                if obj_trajs_full[obj_idx, :, -1].sum() > 0:
                    # Use the last valid timestep as the "current" state
                    valid_times = np.where(obj_trajs_full[obj_idx, :, -1] > 0)[0]
                    last_valid = min(valid_times[-1], current_time_index)
                    center_objects_list.append(obj_trajs_full[obj_idx, last_valid])
                    track_index_to_predict_selected.append(obj_idx)
                    break

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, 10)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict,
            sdc_track_index, timestamps, obj_types, obj_ids
    ):
        """Create centered trajectory data for each center object.

        This mirrors WaymoDataset.create_agent_data_for_center_objects.
        """
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = \
            self.generate_centered_trajs_for_agents(
                center_objects=center_objects, obj_trajs_past=obj_trajs_past,
                obj_types=obj_types, center_indices=track_index_to_predict,
                sdc_index=sdc_track_index, timestamps=timestamps,
                obj_trajs_future=obj_trajs_future
            )

        # Generate labels for center objects
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # Filter invalid past trajectories
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1

        # Generate last valid position for each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask,
                center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
                track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """Transform trajectories to be centered on each center object.

        Same as WaymoDataset.transform_trajs_to_center_coords.
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types,
                                           center_indices, sdc_index, timestamps, obj_trajs_future):
        """Generate centered trajectory features for all agents relative to each center object.

        Mirrors WaymoDataset.generate_centered_trajs_for_agents.

        Args:
            center_objects: (num_center_objects, 10)
            obj_trajs_past: (num_objects, num_timestamps, 10)
            obj_types: (num_objects,)
            center_indices: (num_center_objects,)
            sdc_index: int
            timestamps: (num_timestamps,)
            obj_trajs_future: (num_objects, num_future_timestamps, 10)

        Returns:
            ret_obj_trajs: (num_center_objects, num_objects, num_timestamps, num_attrs)
            ret_obj_valid_mask: (num_center_objects, num_objects, num_timestamps)
            ret_obj_trajs_future: (num_center_objects, num_objects, num_future_timestamps, 4)
            ret_obj_valid_mask_future: (num_center_objects, num_objects, num_future_timestamps)
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape

        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)

        # Transform coordinates to centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # Generate one-hot type masks and other features
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRIAN', :, 1] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # handle original typo
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = torch.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = torch.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        # Generate future trajectory labels
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()

    def collate_batch(self, batch_list):
        """Custom collate that handles pose-specific keys in addition to the standard ones.

        Extends DatasetTemplate.collate_batch.
        """
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                        'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask',
                        'obj_poses', 'obj_poses_mask']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
        return batch_dict

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """Generate prediction dictionaries for evaluation.

        Mirrors WaymoDataset.generate_prediction_dicts.
        """
        input_dict = batch_dict['input_dict']
        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
                }
                cur_scene_pred_list.append(single_pred_dict)
            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', **kwargs):
        """Standalone evaluation for pose dataset — no Waymo TF dependency needed.

        Computes minADE and minFDE for trajectory prediction quality.
        """
        # Flatten nested scene lists
        flat_preds = []
        for entry in pred_dicts:
            if isinstance(entry, list):
                flat_preds.extend(entry)
            else:
                flat_preds.append(entry)

        if len(flat_preds) == 0:
            metric_results = {'minADE': 0.0, 'minFDE': 0.0, 'mAP': 0.0}
            return '\nNo predictions to evaluate.\n', metric_results

        ade_list = []
        fde_list = []
        for pred in flat_preds:
            pred_trajs = pred['pred_trajs']  # (num_modes, num_timestamps, 2)
            gt_trajs_full = pred['gt_trajs']  # (total_timestamps, 10)
            # Future portion starts at num_past (index 11)
            gt_future = gt_trajs_full[self.num_past:, :]  # (80, 10)
            gt_xy = gt_future[:, 0:2]                      # (80, 2)
            gt_valid = gt_future[:, -1] > 0                 # (80,)

            if gt_valid.sum() == 0:
                continue

            # Subsample to match predictions (predictions may be full 80 steps)
            num_pred_steps = pred_trajs.shape[1]
            gt_xy = gt_xy[:num_pred_steps]
            gt_valid = gt_valid[:num_pred_steps]

            # ADE: mean displacement across valid steps, then min over modes
            dist = np.linalg.norm(pred_trajs[:, :, 0:2] - gt_xy[None, :, :], axis=-1)  # (modes, T)
            dist_masked = dist * gt_valid[None, :]
            ade_per_mode = dist_masked.sum(axis=-1) / max(gt_valid.sum(), 1)
            ade_list.append(ade_per_mode.min())

            # FDE: displacement at last valid step, min over modes
            last_valid_idx = np.where(gt_valid)[0]
            if len(last_valid_idx) > 0:
                last_idx = last_valid_idx[-1]
                fde_per_mode = np.linalg.norm(pred_trajs[:, last_idx, 0:2] - gt_xy[last_idx], axis=-1)
                fde_list.append(fde_per_mode.min())

        minADE = float(np.mean(ade_list)) if ade_list else 0.0
        minFDE = float(np.mean(fde_list)) if fde_list else 0.0

        metric_results = {
            'minADE': minADE,
            'minFDE': minFDE,
            'mAP': 0.0,  # placeholder for compatibility with best-model tracking
        }

        metric_result_str = '\n'
        for key, val in metric_results.items():
            metric_result_str += f'{key}: {val:.4f}\n'

        return metric_result_str, metric_results
