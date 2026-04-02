# Research Findings

## Research Question

Does conditioning MTR on past SMPL body poses improve pedestrian trajectory prediction, and what is the optimal pose-trajectory joint learning strategy?

## Current Understanding

The codebase extends Motion Transformer (MTR) with SMPL body pose prediction for pedestrians in the Waymo Open Motion Dataset. The architecture adds:
- A GRU-based pose encoder (input: 144D 6D-rotation poses, output: 256D hidden state)
- MLP fusion of trajectory and pose features
- Per-layer pose prediction heads (144D = 24 joints × 6D) with classification heads
- Losses: MPJPE, geodesic distance on rotations, pose GMM NLL (winner-takes-all L1), pose classification

**Critical bug found on 2026-04-01**: The `_load_pedestrian` function in `waymo_pose_dataset.py` compares raw waymo timestamps (in microseconds, e.g., 100020) against the time grid (in seconds, 0.0 to 9.0). The argmin always maps to index 90 (9.0s) for all non-zero timestamps. Fix: divide timestamps by 1e6 before comparison.

## Key Results

| Run | Model | Best minADE | Best Epoch | Notes |
|-----|-------|-------------|------------|-------|
| H1 | Full pose (GRU + all losses) | 0.6114 | 1 | ⚠ unreliable metric — 8 val samples |
| H2 | Trajectory-only baseline | 0.7724 | 5 | ⚠ unreliable metric — 8 val samples |

**Preliminary signal**: H1 better than H2 by 21% (0.6114 vs 0.7724), but the metric is based on only 8 validation samples and must be verified with a proper evaluation setup.

## RESOLVED (2026-04-02): Data Fix — Real Waymo Future Trajectories Integrated

**Problem**: `final_10fps` SMPL data had observations ONLY in the past window (grid indices 0-10 = 0.0-1.0s). Future window always zero → training loss=0, eval over 8 samples only. Previous H1/H2 results (minADE 0.61 vs 0.77) were meaningless.

**Solution found**: `/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_{split}/sample_{scene_id}_0.pkl` files contain **real 91-step Waymo tracking trajectories** for ALL agents in each scene. Shape: `(num_agents, 91, 10)` — full past + future ground truth.

**Join key**: SMPL file stem (int) == Waymo `object_id` (int) in `track_infos`. No ambiguity.

**Coverage**: 82% of validation pedestrians have real future data. Average 59/80 future timesteps valid. Training set similar.

**Implementation**: `WaymoPoseDataset._load_scene_waymo_trajs()` loads the PKL; `_load_pedestrian()` accepts an optional `waymo_traj` argument. When matched, the real 91-step trajectory replaces the SMPL-derived trajectory. Committed in `6b5084d`.

**H1_v2 training running** with real data: epoch 1 shows `loss=1138→161`, `ade=0.95→0.76` (non-zero!). First reliable experiment.

## Patterns and Insights

- **H1 better than H2 despite no gradient signal**: Both models get zero trajectory loss during training, yet H1 achieves lower minADE. This likely means the pose conditioning helps the model's initialization/early learning before gradients collapse to zero. The 8-sample metric makes this observation unreliable.
- **Loss collapses to 0**: By epoch 7, training loss → 0.000 for H2, consistent with no valid future masks. The small nonzero losses seen occasionally (3.5, 15.9) correspond to the rare batches that include one of the 8 pedestrians with future data.
- **Both models degrade after initial epoch**: The best metric always occurs at epoch 0 or 1 (before the model has overfit to nothing). LR decay steps (epochs 10, 20, 25) cause further instability.

## Lessons and Constraints

- **Timestamp units**: `waymo_timestamps` in .npz files are in microseconds. Must divide by 1e6 before comparing to the Waymo time grid (seconds). Bug is in `_load_pedestrian()` at line 219.
- **Data coverage**: 579 training scenes, each with a few pedestrians (~1-5). Each pedestrian is observed for a short window (7 timesteps in the sample = 0.6s). The alignment to the 91-step Waymo grid means most poses will be zero/invalid — this is expected.
- **GPU**: RTX 4090 (24GB). With batch_size=10 and 80 future frames × 6 modes × 144 pose dims, memory could be tight. Monitor during first run.
- **No HD map**: The pose dataset has no HD map features. The encoder uses empty placeholder map polylines (2 × 20 × 9 zeros). Map attention is still computed but over zero features — verified no NaN in first run.
- **SMPL shapedirs**: `SMPL_NEUTRAL.pkl` has 300-dim shape space (not 10). Must truncate `th_shapedirs[:,:,:10]` and `th_betas[:,:10]` after SMPL_Layer init in decoder. Standard SMPL uses 10 betas; the extended pkl is for research variants.
- **Dataset indexing**: Some scenes have pedestrians with future-only validity (timestamps only in future window). The fallback in `get_interested_agents` must check for valid PAST timestamps, not any valid timestamp. Otherwise `valid_past_mask` filters them out → index -1.
- **BatchNorm-1**: With variable center objects per scene, sometimes a batch has only 1 total center object (across all scenes in the batch). BatchNorm1d fails with size-1 input. Fix: `DATALOADER_DROP_LAST: True`.
- **Pipeline verified**: Full training pipeline works end-to-end at batch_size=10, ~80s/epoch.
- **NaN from gated zero losses**: When pose loss weights are 0.0, do NOT just multiply by weight — IEEE 754 `0 * NaN = NaN`. Must completely skip the loss computation. Fixed by `compute_pose_losses` flag in `get_decoder_loss`.
- **final_10fps data is past-only**: SMPL timestamps cover 0-1.0s (grid indices 0-10). No future observations → training loss=0, eval over 8 samples only. The `final_30fps` directory has full 91-step trajectories (SMPL-simulated future). The `waymo_motion_extracted/` may have real Waymo future trajectories.
- **mAP is hardcoded 0.0**: `generate_prediction_dicts` hardcodes `'mAP': 0.0` as a compatibility placeholder. Cannot use mAP as a metric; use minADE only.
- **NaN divergence from pose losses**: With all pose loss weights at 1.0 (mpjpe, geo, cls_pose, gmm_pose), the model diverges to NaN at epoch 3. Root cause: gradient explosion through SMPL forward pass with diverging 6D rotations; also `geodesic_distance_6d` in loss_utils clamped to exactly ±1.0 giving infinite gradients at boundary. Fix: (1) reduce pose loss weights to 0.1, (2) clamp to `[-1+1e-7, 1-1e-7]`. With these fixes, training is stable through 30 epochs.
- **YAML duplicate keys**: Python's YAML loader silently uses the last value for duplicate keys. The mtr+pose_data.yaml had duplicate DECAY_STEP_LIST, GRAD_NORM_CLIP, LR_CLIP — the last value was used. Cleaned up.

## Open Questions

1. ~~After fixing the timestamp bug, does the training run converge?~~ Pipeline confirmed working.
2. ~~What is the trajectory-only baseline minADE?~~ H2: 0.7724 (but unreliable — 8 samples).
3. ~~How to get real future trajectory labels?~~ SOLVED: processed_scenarios PKL files contain real 91-step Waymo tracking trajectories. 80% coverage, avg 52-59/80 future steps valid.
4. **[ACTIVE] Does pose conditioning help with REAL trajectory GT?** — H1_v2 vs H2_v2 comparison in progress. Expected to complete in ~90 minutes.
5. Is 579 training scenes enough to learn meaningful pose representations? The original MTR used ~486k scenarios. Our subset may be too small for robust pose conditioning.
6. Are the high initial losses (1138 at iter 0) due to pose losses (mpjpe, geo, cls_pose, gmm_pose) dominating? Should loss weights be tuned?

## Optimization Trajectory

| Run | Hypothesis | Best minADE | Best Epoch | Delta vs baseline | Notes |
|-----|-----------|-------------|------------|-------------------|-------|
| run_002 | H2 (no pose) | 0.7724 | 5 | — (baseline) | ⚠ INVALID — 8 val samples |
| run_001 | H1 (full pose) | 0.6114 | 1 | -21% | ⚠ INVALID — 8 val samples |
| run_003 | H1_v2 (full pose, real GT) | 0.7682 | 7 | — | Running, stable (fixed NaN) |
| run_004 | H2_v2 (no pose, real GT) | TBD | — | — | Queued after H1_v2 |
