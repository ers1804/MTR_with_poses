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

*No training runs complete yet. Bootstrap phase.*

## Patterns and Insights

*To be filled after first experiments.*

## Lessons and Constraints

- **Timestamp units**: `waymo_timestamps` in .npz files are in microseconds. Must divide by 1e6 before comparing to the Waymo time grid (seconds). Bug is in `_load_pedestrian()` at line 219.
- **Data coverage**: 579 training scenes, each with a few pedestrians (~1-5). Each pedestrian is observed for a short window (7 timesteps in the sample = 0.6s). The alignment to the 91-step Waymo grid means most poses will be zero/invalid — this is expected.
- **GPU**: RTX 4090 (24GB). With batch_size=10 and 80 future frames × 6 modes × 144 pose dims, memory could be tight. Monitor during first run.
- **No HD map**: The pose dataset has no HD map features. The encoder uses empty placeholder map polylines (2 × 20 × 9 zeros). Map attention is still computed but over zero features — verified no NaN in first run.
- **SMPL shapedirs**: `SMPL_NEUTRAL.pkl` has 300-dim shape space (not 10). Must truncate `th_shapedirs[:,:,:10]` and `th_betas[:,:10]` after SMPL_Layer init in decoder. Standard SMPL uses 10 betas; the extended pkl is for research variants.
- **Dataset indexing**: Some scenes have pedestrians with future-only validity (timestamps only in future window). The fallback in `get_interested_agents` must check for valid PAST timestamps, not any valid timestamp. Otherwise `valid_past_mask` filters them out → index -1.
- **BatchNorm-1**: With variable center objects per scene, sometimes a batch has only 1 total center object (across all scenes in the batch). BatchNorm1d fails with size-1 input. Fix: `DATALOADER_DROP_LAST: True`.
- **Pipeline verified**: Full training pipeline works end-to-end at batch_size=10, ~80s/epoch.

## Open Questions

1. ~~After fixing the timestamp bug, does the training run converge?~~ Pipeline confirmed working, H1 training in progress.
2. What is the trajectory-only baseline minADE for pedestrians?
3. Does pose conditioning help or hurt? (The pose data may be noisy from SMPL fitting)
4. Is 579 scenes enough to learn meaningful pose representations? The original MTR used ~486k scenarios.

## Optimization Trajectory

| Run | Hypothesis | minADE_ped | Delta | Notes |
|-----|-----------|-----------|-------|-------|
| — | — | — | — | No runs yet |
