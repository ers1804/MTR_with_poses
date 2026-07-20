"""Unit test for rotate_root_6d (review item 2.4, agent-centric pose rotation).

Verifies that rotating the root-orientation 6D channel by a center object's
heading (a) equals the matrix identity Rz(-h) @ R on the recovered rotation,
(b) leaves body-joint dims (6:144) untouched, (c) is the identity for h = 0,
and (d) preserves zero rows (padding stays padding).
"""
import numpy as np

from mtr.datasets.waymo.waymo_pose_dataset import (
    rotate_root_6d, axis_angle_to_rotation_matrix, rotation_matrix_to_6d)


def _rz(h):
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _6d_to_cols(d6):
    return d6[0:3], d6[3:6]


def test_rotation_matches_matrix_identity():
    rng = np.random.RandomState(0)
    # random valid root rotation from a random axis-angle
    aa = rng.randn(1, 3).astype(np.float32)
    R = axis_angle_to_rotation_matrix(aa)[0]              # (3,3)
    d6 = rotation_matrix_to_6d(R[None])[0]                # (6,)
    heading = 0.7
    poses = np.zeros((1, 2, 144), dtype=np.float32)       # (C=1, T=2, 144)
    poses[0, :, 0:6] = d6
    poses[0, :, 6:12] = 0.5                               # body-joint sentinel
    out = rotate_root_6d(poses.copy(), np.array([heading]))
    # expected: columns of Rz(-h) @ R
    Rexp = _rz(-heading) @ R.astype(np.float64)
    c1, c2 = _6d_to_cols(out[0, 0])
    assert np.allclose(c1, Rexp[:, 0], atol=1e-5)
    assert np.allclose(c2, Rexp[:, 1], atol=1e-5)
    # body joints untouched
    assert np.allclose(out[0, :, 6:12], 0.5)


def test_identity_for_zero_heading_and_zero_rows():
    rng = np.random.RandomState(1)
    poses = rng.randn(3, 4, 144).astype(np.float32)
    poses[1] = 0.0                                        # padding row
    out = rotate_root_6d(poses.copy(), np.array([0.0, 1.3, -2.1]))
    assert np.allclose(out[0], poses[0], atol=1e-6)       # h=0 => identity
    assert np.abs(out[1]).sum() == 0.0                    # zeros stay zeros
    # norms of the two root 3-vectors preserved (rotation is orthogonal)
    for col in (0, 3):
        a = np.linalg.norm(poses[2, :, col:col+3], axis=-1)
        b = np.linalg.norm(out[2, :, col:col+3], axis=-1)
        assert np.allclose(a, b, atol=1e-5)


def test_per_center_independent_rotations():
    """Each center object gets its own rotation angle."""
    d6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)   # identity rotation
    poses = np.zeros((2, 1, 144), dtype=np.float32)
    poses[:, :, 0:6] = d6
    out = rotate_root_6d(poses.copy(), np.array([0.0, np.pi / 2]))
    assert np.allclose(out[0, 0, 0:6], d6, atol=1e-6)     # center 0: unchanged
    Rexp = _rz(-np.pi / 2)                                # center 1: rotated
    assert np.allclose(out[1, 0, 0:3], Rexp[:, 0], atol=1e-6)
    assert np.allclose(out[1, 0, 3:6], Rexp[:, 1], atol=1e-6)
