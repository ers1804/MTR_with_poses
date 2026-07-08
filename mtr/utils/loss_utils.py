# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 
import torch.nn.functional as F


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs


def nll_loss_pose_gmm(
    pred_poses,           # (batch_size, num_modes, seq_len, pose_dim)
    gt_poses,             # (batch_size, seq_len, pose_dim)
    pred_scores,          # (batch_size, num_modes)
    gt_valid_mask,        # (batch_size, seq_len)
    pre_nearest_mode_idxs=None,
):
    """
    Winner-takes-all pose regression loss.

    Selects the nearest mode by L2 distance on 6D rotation representations,
    then computes L1 loss on the selected mode, masked by valid timesteps.

    Args:
        pred_poses: (B, M, T, D) predicted poses in 6D rotation (D=144 for 24 joints)
        gt_poses: (B, T, D) ground truth poses in 6D rotation
        pred_scores: (B, M) mode scores (unused for mode selection, kept for interface)
        gt_valid_mask: (B, T) validity mask per timestamp
        pre_nearest_mode_idxs: (B,) pre-computed nearest mode indices (optional)

    Returns:
        reg_loss: (B,) regression loss per sample
        nearest_mode_idxs: (B,) index of nearest mode for each sample
    """
    batch_size = pred_scores.shape[0]

    # ===== 1. MODE SELECTION =====
    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        # L2 distance on 6D representations per timestep
        dist = (pred_poses - gt_poses[:, None, :, :]).norm(dim=-1)  # (B, M, T)
        dist = (dist * gt_valid_mask[:, None, :].float()).sum(dim=-1)  # (B, M)
        nearest_mode_idxs = dist.argmin(dim=-1)  # (B,)

    # ===== 2. EXTRACT NEAREST MODE =====
    batch_idxs = torch.arange(batch_size, device=pred_poses.device)
    nearest_poses = pred_poses[batch_idxs, nearest_mode_idxs]  # (B, T, D)

    # ===== 3. L1 REGRESSION LOSS =====
    # Normalize by the number of VALID steps per agent (masking + normalization, P1.1).
    # An agent with zero valid future-pose steps therefore contributes exactly 0 and is
    # excluded from its own denominator (clamp keeps the division finite).
    gt_valid_mask_f = gt_valid_mask.float()
    residual = (gt_poses - nearest_poses).abs()  # (B, T, D)
    residual = residual.mean(dim=-1)  # average over pose dimensions: (B, T)
    denom = gt_valid_mask_f.sum(dim=-1).clamp(min=1.0)  # (B,)
    reg_loss = (residual * gt_valid_mask_f).sum(dim=-1) / denom  # (B,)

    return reg_loss, nearest_mode_idxs


# ===== OPTIONAL: Geodesic Distance Helper =====

def geodesic_distance_6d(pred_6d, gt_6d, reduction='none'):
    """
    Compute geodesic distance on SO(3) from 6D rotation representations.
    
    Args:
        pred_6d: (..., 6) 6D rotation representation
        gt_6d: (..., 6) ground truth
        
    Returns:
        distance: (...) geodesic distance in radians
    """
    # Convert 6D to rotation matrix
    pred_mat = gram_schmidt_orthogonalization(pred_6d)  # (..., 3, 3)
    gt_mat = gram_schmidt_orthogonalization(gt_6d)       # (..., 3, 3)
    
    # Compute relative rotation: R = pred_mat^T @ gt_mat
    rel_rot = torch.einsum('...ij,...jk->...ik', pred_mat.transpose(-1, -2), gt_mat)
    
    # Trace-based geodesic distance: θ = arccos((trace(R) - 1) / 2)
    trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability (avoid inf grad at boundary)
    angle = torch.acos(cos_angle)

    return angle


# ===== Masked pose-supervision losses (future-pose validity mask, action plan P1.1) =====
# Every future-pose loss must be averaged over VALID steps only. On 10fps data ~99.99%
# of future pose target rows are all-zero, so an unmasked loss (the old MPJPE and the
# inert geodesic) trains predictions toward the zero/T-pose instead of measuring
# supervision. `valid_mask` is `center_gt_poses_mask` = pose-row-nonzero AND traj-valid.


def masked_mpjpe(pred_joints, gt_joints, valid_mask):
    """Per-agent Mean Per-Joint Position Error over VALID future-pose steps only.

    Args:
        pred_joints: (B, T, J, 3)
        gt_joints:   (B, T, J, 3)
        valid_mask:  (B, T) bool/float — 1 where the future-pose step is valid.
    Returns:
        (B,) per-agent loss. An agent with zero valid steps contributes exactly 0
        (masked steps get exactly-zero gradient w.r.t. the prediction).
    """
    per_step = F.l1_loss(pred_joints, gt_joints, reduction='none').mean(dim=(-1, -2))  # (B, T)
    valid = valid_mask.to(per_step.dtype)
    denom = valid.sum(dim=-1).clamp(min=1.0)  # (B,)
    return (per_step * valid).sum(dim=-1) / denom  # (B,)


def masked_geodesic_6d(pred_6d, gt_6d, valid_mask):
    """Per-agent SO(3) geodesic distance over VALID future-pose steps only.

    Args:
        pred_6d: (B, T, J, 6)
        gt_6d:   (B, T, J, 6)
        valid_mask: (B, T) bool/float.
    Returns:
        (B,) per-agent loss; zero-valid agents contribute exactly 0.
    """
    ang = geodesic_distance_6d(pred_6d, gt_6d)  # (B, T, J)
    per_step = ang.mean(dim=-1)  # (B, T)
    valid = valid_mask.to(per_step.dtype)
    denom = valid.sum(dim=-1).clamp(min=1.0)  # (B,)
    return (per_step * valid).sum(dim=-1) / denom  # (B,)


def gram_schmidt_orthogonalization(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.
    
    Args:
        x: (..., 6) where first 3 are col1, last 3 are col2
    
    Returns:
        rotation matrix: (..., 3, 3)
    """
    # Extract two column vectors
    col1 = F.normalize(x[..., :3], dim=-1)  # (..., 3)
    col2 = x[..., 3:]  # (..., 3)
    
    # Gram-Schmidt: orthogonalize col2 w.r.t. col1
    col2 = col2 - (col2 * col1).sum(dim=-1, keepdim=True) * col1
    col2 = F.normalize(col2, dim=-1)
    
    # Third column from cross product
    col3 = torch.cross(col1, col2, dim=-1)  # (..., 3)
    
    # Stack into matrix
    rot_mat = torch.stack([col1, col2, col3], dim=-1)  # (..., 3, 3)
    return rot_mat