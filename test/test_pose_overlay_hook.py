"""The AVO overlay hook must reproduce built-in behaviour bit-for-bit for the seed overlay."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
SEED_OVERLAY_DIR = '/home/erik/ssd2/gitprojects/avo_harness/avo/workspace/src'


def _fresh_loader():
    import mtr.utils.pose_overlay as po
    po._cache = None
    po._loaded_from = None
    return po


def test_no_env_no_overlay():
    os.environ.pop('POSE_OVERLAY_DIR', None)
    po = _fresh_loader()
    assert po.get_overlay() is None


def test_seed_overlay_matches_builtin():
    from mtr.datasets.waymo.waymo_pose_dataset import derive_pose_features
    os.environ['POSE_OVERLAY_DIR'] = SEED_OVERLAY_DIR
    po = _fresh_loader()
    ov = po.get_overlay()
    assert ov.MODE == 'early' and ov.FEATURE_DIM == 8
    rng = np.random.RandomState(7)
    p = rng.randn(3, 5, 11, 144).astype(np.float32)
    h = rng.randn(3, 5, 11).astype(np.float32)
    a = derive_pose_features(p.copy(), h.copy())
    b = ov.derive_features(p.copy(), h.copy())
    assert a.shape == b.shape == (3, 5, 11, 8)
    assert np.array_equal(a, b), 'seed overlay must be bit-identical to the built-in'
    os.environ.pop('POSE_OVERLAY_DIR', None)


def test_contract_validation():
    import tempfile, pathlib, pytest
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, 'overlay.py').write_text('FEATURE_DIM = 4\nMODE = "bogus"\n'
                                                 'def derive_features(p, h):\n    return p\n')
        os.environ['POSE_OVERLAY_DIR'] = d
        po = _fresh_loader()
        with pytest.raises(ValueError):
            po.get_overlay()
    os.environ.pop('POSE_OVERLAY_DIR', None)
