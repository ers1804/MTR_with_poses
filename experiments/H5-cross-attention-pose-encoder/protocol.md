# H5: Cross-Attention Pose Encoder

## Hypothesis

Replacing the GRU+MLP pose encoder with a cross-attention mechanism will improve trajectory prediction further beyond the geo_only GRU baseline (0.6231 minADE), because cross-attention allows the model to attend to specific gait phases (timesteps) rather than compressing the full 11-frame sequence into a single vector.

## Motivation

The current architecture:
1. GRU processes 11 past poses (T=11, 144D each) → compresses to one 256D hidden state
2. MLP fuses [trajectory_feat (256D), pose_hidden (256D)] → 256D output per agent

The bottleneck: GRU must compress 11×144=1584 values into 256D. It cannot differentially weight timesteps — all past frames contribute equally to the final hidden state via the sequential GRU update. This is a lossy aggregation.

The cross-attention alternative:
1. Project each pose frame to D_MODEL (11 tokens × 256D)
2. The agent feature (from PointNet polyline encoder) queries these 11 pose tokens
3. The model learns which gait phase timesteps are most predictive of future trajectory
4. Residual addition or MLP fusion of attended pose context with agent feature

**Why this might work better**:
- Recent timestep poses likely matter more than early ones for direction prediction
- Foot plant vs mid-swing phases carry different trajectory-relevant information
- GRU gradients suffer from vanishing gradient over 11 steps (though 11 is short)
- Cross-attention with geodesic supervision should create sharper gradient signal to specific timestep features

## Prediction

- **Strong**: Cross-attention improves over GRU geo_only (0.6231 → <0.62)
- **Weak**: Cross-attention gives similar performance to GRU (within 0.5%), meaning GRU already saturates the useful pose information at this scale
- **If weak**: The mechanism is already saturated — 11 frames is short enough for GRU to work well

## Implementation Plan

### Architecture Change

In `mtr/models/context_encoder/mtr_encoder.py`:

Replace:
```python
self.pose_encoder = torch.nn.GRU(...)  # 11×144 → 256
self.pose_fuser = MLP(in=512, out=256)  # [traj(256), pose(256)] → 256
```

With:
```python
self.pose_proj = nn.Linear(144, D_MODEL)  # project each frame to D_MODEL
self.pose_cross_attn = nn.MultiheadAttention(
    embed_dim=D_MODEL, num_heads=8, dropout=0.1, batch_first=True
)
self.pose_fuser = MLP(in=512, out=256)   # same fusion as before
```

Forward:
```python
# obj_poses: (num_center_objects, num_objects, T=11, 144)
B, N, T, C = obj_poses.shape
poses_flat = self.pose_proj(obj_poses.reshape(B*N, T, C))  # (B*N, T, D_MODEL)
agent_query = obj_polylines_feature.reshape(B*N, 1, D_MODEL)  # (B*N, 1, D_MODEL)
key_padding_mask = ~combined_mask.reshape(B*N, T)  # True = ignore
pose_ctx, _ = self.pose_cross_attn(
    query=agent_query, key=poses_flat, value=poses_flat,
    key_padding_mask=key_padding_mask
)  # (B*N, 1, D_MODEL)
pose_ctx = pose_ctx.squeeze(1).reshape(B, N, D_MODEL)
# zero out agents without valid poses
pose_ctx[~combined_valid_mask] = 0.0
fused = torch.cat([obj_polylines_feature, pose_ctx], dim=-1)
obj_polylines_feature = self.pose_fuser(fused)
```

### Config

New config: `tools/cfgs/waymo/mtr+pose_data_cross_attn.yaml`
- Same as `mtr+pose_data_geo_only.yaml` except:
  - Replace `NUM_LAYER_IN_POSE_GRU`, `DROPOUT_OF_POSE_GRU` with `USE_CROSS_ATTN_POSE: True`
  - Same loss weights: geo=0.1, gmm_pose=0.1, others=0.0

### Run Tag

`H5_cross_attn_geo_only`

## Expected Timeline

~13 minutes (same as other 30-epoch runs)

## Success Criteria

- Training converges without NaN
- Best minADE < 0.6231 (beat GRU geo_only)
- If NaN occurs: debug cross-attn key_padding_mask (all-masked agents)
