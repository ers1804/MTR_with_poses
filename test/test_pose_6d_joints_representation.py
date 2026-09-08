"""The '6d+joints' representation must expose articulation WITHOUT world/agent yaw."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def _ds(rep):
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    d = WaymoPoseDataset.__new__(WaymoPoseDataset)
    d.pose_representation = rep
    d._smpl_layer = None
    d.dataset_cfg = _Cfg(SMPL_MODEL_DIR='/home/erik/ssd2/gitprojects/MTR_with_poses/models')
    return d


def test_dims_and_layout():
    d = _ds('6d+joints')
    rng = np.random.RandomState(0)
    root = rng.randn(4, 3).astype(np.float32) * 0.3
    body = rng.randn(4, 69).astype(np.float32) * 0.1
    betas = np.zeros(10, dtype=np.float32)
    f = d._pose_features(root, body, betas)
    assert f.shape == (4, 216), f.shape
    assert np.isfinite(f).all()
    # first 144 dims must equal the plain 6D representation
    from mtr.datasets.waymo.waymo_pose_dataset import smpl_params_to_6d
    assert np.allclose(f[:, :144], smpl_params_to_6d(root, body), atol=1e-6)


def test_joint_block_is_yaw_invariant():
    """Rotating the body in the world must not change the position block."""
    d = _ds('6d+joints')
    rng = np.random.RandomState(1)
    body = rng.randn(3, 69).astype(np.float32) * 0.1
    betas = np.zeros(10, dtype=np.float32)
    base = d._pose_features(np.zeros((3, 3), np.float32), body, betas)[:, 144:]
    for yaw in (0.5, -1.7, 2.9):
        root = np.tile(np.array([0, 0, yaw], np.float32), (3, 1))   # yaw about world z
        rot = d._pose_features(root, body, betas)[:, 144:]
        assert np.allclose(base, rot, atol=1e-5), f'position block moved under yaw {yaw}'


def test_joints3d_still_carries_world_rotation():
    """Guard the contrast: the legacy representation is NOT yaw-invariant."""
    d = _ds('joints3d')
    rng = np.random.RandomState(2)
    body = rng.randn(2, 69).astype(np.float32) * 0.1
    betas = np.zeros(10, dtype=np.float32)
    a = d._pose_features(np.zeros((2, 3), np.float32), body, betas)
    b = d._pose_features(np.tile(np.array([0, 0, 1.2], np.float32), (2, 1)), body, betas)
    assert not np.allclose(a, b, atol=1e-3)
