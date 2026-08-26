"""Derived pose descriptors must be invariant to the yaw convention (review 2026-08-26)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mtr.datasets.waymo.waymo_pose_dataset import derive_pose_features, rotate_root_6d


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# The body forward axis is the THIRD column of the root rotation. A pure yaw about world-z would
# leave it pointing at the zenith (yaw undefined), which is not how real SMPL roots look, so the
# fixture first tips the body forward axis into the ground plane (Ry(90 deg)), then yaws it.
_UPRIGHT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)


def _random_pose(T=11, seed=0, yaw0=0.7):
    rng = np.random.RandomState(seed)
    p = np.zeros((1, T, 144), dtype=np.float32)
    for t in range(T):
        R = _rz(yaw0 + 0.05 * t) @ _UPRIGHT
        p[0, t, 0:3], p[0, t, 3:6] = R[:, 0], R[:, 1]
        p[0, t, 6:] = rng.randn(138) * 0.1
    return p


def test_fixture_forward_axis_is_horizontal():
    """Guard the fixture itself: body forward must lie in the ground plane."""
    p = _random_pose()
    fwd = np.cross(p[0, 0, 0:3], p[0, 0, 3:6])
    assert abs(fwd[2]) < 1e-5, "fixture body forward is vertical; yaw would be undefined"
    assert np.allclose(np.arctan2(fwd[1], fwd[0]), 0.7, atol=1e-5)


def test_invariant_to_global_yaw_rotation():
    """Rotating the world frame must not change the descriptors."""
    p = _random_pose()
    head = np.full((1, 11), 0.7, dtype=np.float32)
    base = derive_pose_features(p, head)
    for delta in (0.5, -1.2, 3.0):
        p_rot = rotate_root_6d(p.copy(), np.array([-delta], dtype=np.float32))
        rot = derive_pose_features(p_rot, head + delta)
        assert np.allclose(base, rot, atol=1e-5), f"not invariant under yaw {delta}"


def test_relative_facing_is_recovered():
    """cos/sin channels must encode body-yaw minus travel-heading."""
    p = _random_pose()
    offset = 0.4
    head = np.full((1, 11), 0.7 - offset, dtype=np.float32)   # body leads heading by `offset`
    f = derive_pose_features(p, head)
    assert np.allclose(f[0, 0, 0], np.cos(offset), atol=1e-3)
    assert np.allclose(f[0, 0, 1], np.sin(offset), atol=1e-3)


def test_shape_and_finiteness():
    p = _random_pose(seed=3)
    f = derive_pose_features(p, np.zeros((1, 11), dtype=np.float32))
    assert f.shape == (1, 11, 8)
    assert np.isfinite(f).all()
