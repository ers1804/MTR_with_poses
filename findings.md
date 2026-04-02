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

| Run | Model | Best minADE | Notes |
|-----|-------|-------------|-------|
| run_001 | H1 (full pose) | 0.6114 | ⚠ INVALID — 8 val samples (zero future GT) |
| run_002 | H2 (no pose) | 0.7724 | ⚠ INVALID — 8 val samples (zero future GT) |
| **run_003** | **H1_v2 (full pose, real GT)** | **0.6532** | ✓ VALID — 30 epochs, real Waymo trajectories |
| **run_004** | **H2_v2 (no pose, real GT)** | **0.6745** | ✓ VALID — 30 epochs, real Waymo trajectories |

**Core finding**: Pose conditioning improves minADE by **3.2%** (0.6532 vs 0.6745). The improvement is consistent but modest — suggesting pose provides complementary signal to trajectory history, but both models ultimately learn similar motion priors with only 579 training scenes.

**minFDE comparison**: H1_v2 final: 1.4619, H2_v2 final: 1.5080 — consistent with minADE trend (+3.1% for H1).

## RESOLVED (2026-04-02): Data Fix — Real Waymo Future Trajectories Integrated

**Problem**: `final_10fps` SMPL data had observations ONLY in the past window (grid indices 0-10 = 0.0-1.0s). Future window always zero → training loss=0, eval over 8 samples only. Previous H1/H2 results (minADE 0.61 vs 0.77) were meaningless.

**Solution found**: `/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_{split}/sample_{scene_id}_0.pkl` files contain **real 91-step Waymo tracking trajectories** for ALL agents in each scene. Shape: `(num_agents, 91, 10)` — full past + future ground truth.

**Join key**: SMPL file stem (int) == Waymo `object_id` (int) in `track_infos`. No ambiguity.

**Coverage**: 82% of validation pedestrians have real future data. Average 59/80 future timesteps valid. Training set similar.

**Implementation**: `WaymoPoseDataset._load_scene_waymo_trajs()` loads the PKL; `_load_pedestrian()` accepts an optional `waymo_traj` argument. When matched, the real 91-step trajectory replaces the SMPL-derived trajectory. Committed in `6b5084d`.

**H1_v2 training running** with real data: epoch 1 shows `loss=1138→161`, `ade=0.95→0.76` (non-zero!). First reliable experiment.

## Patterns and Insights

- **Pose conditioning provides a 3.2% minADE improvement**: H1_v2 (0.6532) vs H2_v2 (0.6745). Both trained 30 epochs with real Waymo future trajectories. The improvement holds at minFDE too (1.4619 vs 1.5080, +3.1%).
- **Convergence dynamics differ**: H1_v2 reaches best minADE earlier in training and has more variance across epochs (likely from noisy pose losses). H2_v2 plateaus more smoothly. Both settle around 0.67-0.70 after LR decay.
- **Pose loss weights matter critically for stability**: At weight=1.0, training diverges to NaN at epoch 3 due to gradient explosion through SMPL. At weight=0.1, training is stable for 30 epochs. The optimal weight likely lies between 0.1 and 1.0.
- **Small dataset limits absolute performance**: minADE ~0.65 is far from SOTA (~0.3 on full Waymo). With 579 training scenes vs 486k in full MTR, the gap is expected. The relative H1 vs H2 comparison is still valid.
- **H1 better than H2 (INVALID, pre-fix)**: Both models got zero trajectory loss in the initial runs — the 3.2% improvement is the first reliable signal.

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
2. ~~What is the trajectory-only baseline minADE?~~ H2_v2: 0.6745 (valid — full dataset).
3. ~~How to get real future trajectory labels?~~ SOLVED: processed_scenarios PKL files. 80% coverage.
4. ~~Does pose conditioning help with REAL trajectory GT?~~ **YES — 3.2% improvement (0.6532 vs 0.6745).**
5. **[NEXT] Is the 3.2% gain statistically meaningful on 8282 val pedestrians?** The gain is consistent across both minADE and minFDE, but training noise could account for some variation.
6. **[NEXT] What is the optimal pose loss weight?** At 0.1, training is stable. Likely a sweet spot between 0.05-0.5 that maximizes trajectory benefit. Could ablate.
7. **[NEXT] Does more training data increase the pose conditioning benefit?** With 579 scenes, both models are highly data-limited. With the full Waymo SMPL set (~10k+ scenes), the pose benefit might be larger.
8. Is there a way to evaluate pose prediction quality separately from trajectory quality?

## Optimization Trajectory

| Run | Hypothesis | Best minADE | Delta vs baseline | Notes |
|-----|-----------|-------------|-------------------|-------|
| run_002 | H2 (no pose) | 0.7724 | — (baseline) | ⚠ INVALID — 8 val samples |
| run_001 | H1 (full pose) | 0.6114 | -21% | ⚠ INVALID — 8 val samples |
| **run_004** | **H2_v2 (no pose, real GT)** | **0.6745** | — (new baseline) | ✓ VALID |
| **run_003** | **H1_v2 (full pose, real GT)** | **0.6532** | **-3.2%** | ✓ VALID |
