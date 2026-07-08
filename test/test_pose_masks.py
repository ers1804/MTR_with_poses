"""Gradient tests for the future-pose validity mask (Phase 1, action plan P1.1).

These assert that every pose-supervision loss is masked by a per-step validity
mask so that:
  (a) the gradient w.r.t. predictions at INVALID steps is exactly zero,
  (b) the gradient at VALID steps is nonzero,
  (c) an agent with ZERO valid future-pose steps contributes exactly zero loss.

This is the test that would have caught both the inert-geodesic bug (2026-06-12)
and the still-live unmasked-MPJPE bug: on 10fps data 99.99% of future pose target
rows are all-zero, so an unmasked loss trains predictions toward the zero/T-pose.

Run: pytest test/test_pose_masks.py -q
"""
import torch

from mtr.utils import loss_utils


# Mask pattern shared across tests:
#   agent 0: valid at steps [0, 2], invalid at [1, 3]
#   agent 1: NO valid steps  (zero-valid agent)
B, T, J = 2, 4, 24
VALID = torch.tensor([[1, 0, 1, 0],
                      [0, 0, 0, 0]], dtype=torch.bool)
VALID_STEPS_A0 = [0, 2]
INVALID_STEPS_A0 = [1, 3]


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _assert_step_gradients(grad_per_step, name):
    """grad_per_step: (B, T) L1-norm of the gradient at each (agent, step)."""
    # (a) invalid steps of agent 0 -> exactly zero
    for s in INVALID_STEPS_A0:
        assert grad_per_step[0, s].item() == 0.0, f"{name}: grad leaked at invalid step {s}"
    # (b) valid steps of agent 0 -> nonzero
    for s in VALID_STEPS_A0:
        assert grad_per_step[0, s].item() > 0.0, f"{name}: no grad at valid step {s}"
    # (c) zero-valid agent 1 -> exactly zero everywhere
    assert grad_per_step[1].abs().sum().item() == 0.0, f"{name}: grad leaked on zero-valid agent"


def test_masked_mpjpe_gradients():
    pred = _rand(B, T, J, 3, seed=1).requires_grad_(True)
    gt = _rand(B, T, J, 3, seed=2)
    per_agent = loss_utils.masked_mpjpe(pred, gt, VALID)   # (B,)
    assert per_agent.shape == (B,)
    # (c) zero-valid agent contributes exactly zero
    assert per_agent[1].item() == 0.0
    per_agent.sum().backward()
    grad_per_step = pred.grad.abs().flatten(start_dim=2).sum(dim=-1)  # (B, T)
    _assert_step_gradients(grad_per_step, "mpjpe")


def test_masked_geodesic_gradients():
    pred = _rand(B, T, J, 6, seed=3).requires_grad_(True)
    gt = _rand(B, T, J, 6, seed=4)
    per_agent = loss_utils.masked_geodesic_6d(pred, gt, VALID)  # (B,)
    assert per_agent.shape == (B,)
    assert per_agent[1].item() == 0.0
    per_agent.sum().backward()
    grad_per_step = pred.grad.abs().flatten(start_dim=2).sum(dim=-1)  # (B, T)
    _assert_step_gradients(grad_per_step, "geodesic")


def test_masked_wta_l1_gradients():
    M, D = 6, 144
    pred = _rand(B, M, T, D, seed=5).requires_grad_(True)
    gt = _rand(B, T, D, seed=6)
    scores = _rand(B, M, seed=7)
    reg_loss, mode_idxs = loss_utils.nll_loss_pose_gmm(
        pred_poses=pred, gt_poses=gt, pred_scores=scores, gt_valid_mask=VALID,
    )
    assert reg_loss.shape == (B,)
    # (c) zero-valid agent contributes exactly zero
    assert reg_loss[1].item() == 0.0
    reg_loss.sum().backward()
    # gradient only flows to the winning mode; sum over modes to get per-step norm
    grad_per_step = pred.grad.abs().sum(dim=1).flatten(start_dim=2).sum(dim=-1)  # (B, T)
    _assert_step_gradients(grad_per_step, "wta_l1")


def test_zero_valid_agent_is_dropped_from_normalization():
    """A fully-invalid agent must not dilute the per-step normalization of a valid one."""
    pred = _rand(B, T, J, 3, seed=8).requires_grad_(True)
    gt = _rand(B, T, J, 3, seed=9)
    per_agent = loss_utils.masked_mpjpe(pred, gt, VALID)
    # agent 0 loss == plain mean of the two valid-step per-joint L1s (denominator = 2, not 4)
    per_step = torch.nn.functional.l1_loss(pred, gt, reduction='none').mean(dim=(-1, -2))  # (B,T)
    expected_a0 = per_step[0, VALID_STEPS_A0].mean()
    assert torch.allclose(per_agent[0], expected_a0, atol=1e-6)
