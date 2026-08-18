"""Unit tests for denoise_pose_sequence (Part 7 pre-registered experiment)."""
import numpy as np
from mtr.datasets.waymo.waymo_pose_dataset import denoise_pose_sequence


def test_masks_and_padding_untouched():
    rng = np.random.RandomState(0)
    seq = rng.randn(11, 144).astype(np.float32)
    seq[3] = 0.0; seq[7] = 0.0                      # padding rows
    out = denoise_pose_sequence(seq.copy())
    assert np.abs(out[3]).sum() == 0 and np.abs(out[7]).sum() == 0
    valid_before = np.abs(seq).sum(-1) > 0
    valid_after = np.abs(out).sum(-1) > 0
    assert np.array_equal(valid_before, valid_after)


def test_reduces_jitter_keeps_signal():
    t = np.arange(11)
    smooth = np.sin(t / 3.0)[:, None] * np.ones((1, 144))
    noise = np.random.RandomState(1).randn(11, 144) * 0.3
    seq = (smooth + noise).astype(np.float32)
    out = denoise_pose_sequence(seq.copy())
    jit_in = np.linalg.norm(np.diff(seq, axis=0), axis=-1).mean()
    jit_out = np.linalg.norm(np.diff(out, axis=0), axis=-1).mean()
    assert jit_out < 0.75 * jit_in                   # jitter reduced
    err_in = np.linalg.norm(seq - smooth, axis=-1).mean()
    err_out = np.linalg.norm(out - smooth, axis=-1).mean()
    assert err_out < err_in                          # closer to the clean signal


def test_single_valid_frame_identity():
    seq = np.zeros((11, 144), dtype=np.float32)
    seq[5] = 1.0
    out = denoise_pose_sequence(seq.copy())
    assert np.allclose(out, seq)
