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
                          # pose_dim = 3 (root_pos) + num_joints*6 (6D rot)
    gt_poses,             # (batch_size, seq_len, pose_dim)
    pred_scores,          # (batch_size, num_modes)
    gt_valid_mask,        # (batch_size, seq_len)
    use_square_gmm=False,
    log_std_range=(-10.0, 3.0),
    rho_limit=0.99,
    timestamp_loss_weight=None,
    pose_weight_components=None,  # NEW: weight for root vs joints
    pre_nearest_mode_idxs=None,
):
    """
    NLL loss for SMPL pose prediction with mixture of Gaussians.
    
    Args:
        pred_poses: (B, M, T, D) where D = 3 (root) + J*6 (6D rotations per joint)
        gt_poses: (B, T, D)
        pred_scores: (B, M) - mixing coefficients
        gt_valid_mask: (B, T) - validity mask per timestamp
        use_square_gmm: if True, use diagonal GMM (std1 == std2, rho=0)
        pose_weight_components: tuple of (root_weight, joint_weight) for distance metric
        pre_nearest_mode_idxs: (B,) pre-computed nearest mode indices (optional)
    
    Returns:
        reg_loss: (B,) - negative log likelihood per sample
        nearest_mode_idxs: (B,) - index of nearest mode for each batch
    """
    
    batch_size = pred_scores.shape[0]
    num_modes = pred_poses.shape[1]
    seq_len = pred_poses.shape[2]
    
    # Default pose component weighting (can be tuned)
    if pose_weight_components is None:
        pose_weight_components = (1.0, 1.0)  # (root_weight, joint_weight)
    
    root_weight, joint_weight = pose_weight_components
    
    # ===== 1. COMPUTE POSE DISTANCE (Multimodal Mode Selection) =====
    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        # Split poses into root position and joint rotations
        # pred_root = pred_poses[:, :, :, :3]          # (B, M, T, 3)
        # gt_root = gt_poses[:, :, :3].unsqueeze(1)    # (B, 1, T, 3)
        
        pred_joints = pred_poses        # (B, M, T, J*6)
        gt_joints = gt_poses.unsqueeze(1)  # (B, 1, T, J*6)
        
        # Root position distance (Euclidean)
        #root_dist = (pred_root - gt_root).norm(dim=-1)  # (B, M, T)
        
        # Joint rotation distance
        # Option A: L2 distance on 6D representation (simple, differentiable)
        joint_dist = (pred_joints - gt_joints).norm(dim=-1)  # (B, M, T)
        
        # Option B: Geodesic distance on SO(3) (more theoretically sound)
        # Uncomment if you want to convert 6D to rotation matrix first
        # See geodesic_distance_6d() function below
        geodesic_distance_6d(pred_joints, gt_joints, reduction='none')  # (B, M, T)
        
        # Weighted combination
        pose_distance = joint_weight * joint_dist  # (B, M, T)
        
        # Apply validity mask and sum over timestamps
        pose_distance = (pose_distance * gt_valid_mask[:, None, :]).sum(dim=-1)  # (B, M)
        
        # Find nearest mode for each batch sample
        nearest_mode_idxs = pose_distance.argmin(dim=-1)  # (B,)
    
    # ===== 2. EXTRACT NEAREST MODE PARAMETERS =====
    batch_idxs = torch.arange(batch_size, device=pred_poses.device, dtype=nearest_mode_idxs.dtype)
    
    nearest_poses = pred_poses[batch_idxs, nearest_mode_idxs]  # (B, T, pose_dim)
    nearest_root = nearest_poses[:, :, :3]                     # (B, T, 3)
    nearest_joints = nearest_poses[:, :, 3:]                   # (B, T, J*6)
    
    gt_root = gt_poses[:, :, :3]
    gt_joints = gt_poses[:, :, 3:]
    
    # Residuals (GT - Predicted)
    res_root = gt_root - nearest_root  # (B, T, 3)
    res_joints = gt_joints - nearest_joints  # (B, T, J*6)
    
    # ===== 3. HANDLE UNCERTAINTY PARAMETERIZATION =====
    
    # For SIMPLE case: model uncertainty as scalar per timestamp
    if use_square_gmm:
        # Diagonal GMM: σ_x = σ_y for root position
        # Single log-std parameter in position [3] of root prediction
        assert pred_poses.shape[-1] >= 4, "Need at least 4 dims for square GMM (3 root + 1 log_std)"
        
        log_std_root = torch.clip(
            nearest_poses[:, :, 3], 
            min=log_std_range[0], 
            max=log_std_range[1]
        )  # (B, T)
        std_root = torch.exp(log_std_root)  # (B, T)
        
        # For joints: could add after position, e.g., position [4]
        # Or model separately. Here we'll use same std for simplicity.
        log_std_joints = log_std_root.clone()
        std_joints = std_root.clone()
        rho = torch.zeros_like(log_std_root)
        
    else:
        # Full bivariate Gaussian per component
        # Expected layout: [3: root_pos] + [1: log_std_x] + [1: log_std_y] + [1: rho]
        # Adjust based on your actual prediction structure
        
        log_std_root_x = torch.clip(
            nearest_poses[:, :, 3],
            min=log_std_range[0],
            max=log_std_range[1]
        )
        log_std_root_y = torch.clip(
            nearest_poses[:, :, 4],
            min=log_std_range[0],
            max=log_std_range[1]
        )
        std_root_x = torch.exp(log_std_root_x)
        std_root_y = torch.exp(log_std_root_y)
        
        rho = torch.clip(nearest_poses[:, :, 5], min=-rho_limit, max=rho_limit)
        
        log_std_joints = torch.clip(
            nearest_poses[:, :, 6] if pred_poses.shape[-1] > 6 else log_std_root_x,
            min=log_std_range[0],
            max=log_std_range[1]
        )
        std_joints = torch.exp(log_std_joints)
    
    # ===== 4. COMPUTE GMM NLL (ROOT POSITION) =====
    gt_valid_mask = gt_valid_mask.type_as(pred_poses)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]
    
    # Root position as bivariate Gaussian
    if use_square_gmm:
        # Diagonal: σ_x = σ_y
        dx = res_root[:, :, 0]  # (B, T)
        dy = res_root[:, :, 1]
        
        log_coeff = 2 * log_std_root  # log(σ^2) for both dimensions
        nll_root = (dx**2 + dy**2) / (std_root**2)
        
    else:
        # Full bivariate Gaussian
        dx = res_root[:, :, 0]
        dy = res_root[:, :, 1]
        
        log_coeff = log_std_root_x + log_std_root_y + 0.5 * torch.log(1 - rho**2)
        
        nll_root = (0.5 / (1 - rho**2)) * (
            (dx**2) / (std_root_x**2) + 
            (dy**2) / (std_root_y**2) - 
            2 * rho * dx * dy / (std_root_x * std_root_y)
        )
    
    reg_loss_root = ((log_coeff + nll_root) * gt_valid_mask).sum(dim=-1)  # (B,)
    
    # ===== 5. COMPUTE GMM NLL (JOINT ROTATIONS) =====
    # Simple approach: treat 6D rotation as Gaussian with per-dimension std
    # More sophisticated: model full covariance, but 6D has structure
    
    # For now: diagonal Gaussian over 6D vector
    nll_joints = (res_joints**2) / (std_joints.unsqueeze(-1)**2)  # (B, T, J*6)
    nll_joints = nll_joints.mean(dim=-1)  # Average across joint dimensions (B, T)
    
    log_coeff_joints = torch.log(std_joints.unsqueeze(-1)).sum(dim=-1)  # (B, T)
    reg_loss_joints = ((log_coeff_joints + nll_joints) * gt_valid_mask).sum(dim=-1)
    
    # ===== 6. COMBINE LOSSES =====
    # Weighted combination of root and joint losses
    total_loss = root_weight * reg_loss_root + joint_weight * reg_loss_joints
    
    return total_loss, nearest_mode_idxs


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
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = torch.acos(cos_angle)
    
    return angle


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