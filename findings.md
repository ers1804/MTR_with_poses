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

| Run | Model | Pose Weight | Data | Best minADE | Δ vs baseline | Notes |
|-----|-------|-------------|------|-------------|---------------|-------|
| run_001 | H1 (full pose) | 1.0 | 10fps | 0.6114 | — | ⚠ INVALID — 8 val samples (zero future GT) |
| run_002 | H2 (no pose) | 0.0 | 10fps | 0.7724 | — | ⚠ INVALID — 8 val samples (zero future GT) |
| **run_004** | **H2_v2 (no pose, real GT)** | 0.0 | 10fps | **0.6745** | — (baseline) | ✓ VALID |
| run_005 | H1_w05 (pose, real GT) | 0.05 | 10fps | 0.6557 | −2.8% | ✓ VALID |
| **run_003** | **H1_v2 (pose, real GT)** | **0.1** | **10fps** | **0.6532** | **−3.2%** | ✓ VALID ← optimal |
| run_006 | H1_w2 (pose, real GT) | 0.2 | 10fps | 0.6542 | −3.0% | ✓ VALID |
| run_007 | H1_v3 (30fps AMASS future) | 0.1 | 30fps | 0.6567 | −2.6% | ✓ VALID — null result |
| run_008 | H3 mpjpe_only | 0.1 | 10fps | 0.6576 | −2.5% | ✓ VALID — joint-space L1 only |
| run_009 | H3 gmm_only | 0.1 | 10fps | 0.6467 | −4.1% | ✓ VALID — WTA NLL only |
| **run_010** | **H3 geo_only** | **0.1** | **10fps** | **0.6231** | **−7.6%** | ✓ VALID ← new best! |
| run_011 | H3 geo_w=0.05 | 0.05 | 10fps | 0.6786 | −1.5% | ✓ VALID — geo signal too weak |
| run_012 | H3 geo_w=0.2 | 0.2 | 10fps | 0.6660 | −1.3% | ✓ VALID — too strong, dominates trajectory |
| run_013 | H5 cross-attn | 0.1 | 10fps | 0.6765 | +0.3% | ✓ VALID — essentially baseline (no improvement) |
| run_014 | H5b cross-attn+PE | 0.1 | 10fps | 0.6337 | −6.0% | ✓ VALID — PE recovers 79% of GRU advantage |
| **run_015** | **H6 no_pose+map** | 0.0 | 10fps | **0.4880** | **−27.7%** | ✓ VALID — HD map is dominant context signal |
| **run_016** | **H6 geo_only+map** | 0.1 | 10fps | **0.4797** | **−28.9% (vs no-map baseline)** | ✓ VALID — best from-scratch result |
| run_017 | H7 full-Waymo pretrain | 0.0 | 10fps (231k peds) | — | — | ✓ VALID — pretrain only, eval not run |
| run_019 | H9 no_pose+map, H7 pretrain → finetune | 0.0 | 10fps | 0.3861 | −42.8% | ✓ VALID — pretrain ablation, no pose |
| **run_018** | **H8 geo+map, H7 pretrain → finetune** | 0.1 | 10fps | **0.3749** | **−44.4%** | ✓ VALID ← NEW OVERALL BEST |

**Core finding**: Pose conditioning improves minADE by **3.2%** (0.6532 vs 0.6745) with full losses at weight=0.1. The improvement is **robust across pose weights [0.05, 0.2]** and across loss ablations (all variants improve over baseline).

**H6 key finding — HD map context dominates; pose benefit shrinks under map**: Adding real HD map polylines to both the no-pose baseline and the best pose model yields a complete 2×2 ablation:

| | No map | With map | Map Δ |
|---|---|---|---|
| No pose | 0.6745 | 0.4880 | −27.7% |
| GRU+geo | 0.6231 | **0.4797** | −23.0% |
| Pose Δ | −7.6% | **−1.7%** | — |

HD map is the dominant context signal — it reduces minADE by ~27% for both conditions. The pose benefit **shrinks from 7.6% to 1.7%** when map features are available. Interpretation: body orientation (pose) and map spatial routing partially encode the same information about where the agent is heading. When map provides explicit routing constraints, pose's implicit orientation signal becomes mostly redundant. Pose still helps (1.7%), but the gain is much smaller than without map.

**H7→H8/H9 key finding — full-Waymo pretraining is the dominant axis; pose adds ~3% on top**: Pretraining the trajectory-only backbone on the full Waymo pedestrian set (231k agent-batch examples across 487k scenes, 30 epochs) and then fine-tuning two variants on the 579-scene subset (H9 = no pose encoder; H8 = pose encoder with residual zero-init `pose_fuser`) gives the clean 2×2 of pretrain × pose:

| | No pose | With pose (geo+gmm) | Pose Δ |
|---|---|---|---|
| **From scratch (H6)** | 0.4880 | 0.4797 | −1.7% |
| **H7 pretrain → finetune** | **0.3861** (H9) | **0.3749** (H8) | **−2.9%** |
| **Pretrain Δ** | **−20.9%** | **−21.9%** | — |

Two clean conclusions:
1. **Pretraining is the dominant lever**: full-Waymo pretraining yields ~21% minADE reduction, *independent* of whether the pose encoder is used. The 579-scene subset was the binding constraint, not architecture.
2. **Pose still helps on the pretrained backbone, slightly more than from scratch**: 2.9% vs 1.7%. Plausible reason: with from-scratch noisy trajectory features, pose's incremental signal gets averaged into general representation learning; with a strong pretrained trajectory prior, pose adds focused orientation/heading information rather than competing for capacity.

The residual+zero-init `pose_fuser` (`POSE_FUSER_RESIDUAL: True` in config) is what makes the warm-start clean: at fine-tune iter 0 the pose pathway adds zero, so the pretrained backbone passes through unchanged, and the pose encoder warms up additively without corrupting trajectory features.

**H3 key finding — geodesic loss drives the benefit**: Ablating pose losses individually reveals geodesic distance (rotation-space supervision) is by far the most powerful component:
- geo_only: **0.6231 (−7.6%)** — the new best result, better than the full model
- gmm_only: 0.6467 (−4.1%)
- mpjpe_only: 0.6576 (−2.5%)
- full model: 0.6532 (−3.2%)

**Counterintuitive finding**: The full model (all 4 losses combined) performs WORSE than geo_only. MPJPE (joint-space) supervision appears to conflict with geodesic (rotation-space) supervision, degrading the shared GRU encoder features. Geodesic loss in SO(3) is a more natural supervisory signal for body poses because it directly penalizes rotation errors without the nonlinear FK transformation required for MPJPE.

**H5/H5b architecture ablation — temporal ordering dominates, sequential integration adds incrementally**:
- H5 cross-attn (no PE): 0.6765 — essentially matches baseline (0.6745). Bag-of-items failure: without temporal ordering, cross-attention cannot learn gait dynamics from geodesic gradients.
- H5b cross-attn + sinusoidal PE: **0.6337** — PE restores temporal ordering and recovers **79% of GRU's advantage** (0.0408 of 0.0514 total gain vs baseline).
- GRU geo_only: 0.6231 — sequential hidden state integration adds a further **1.6% gain** on top of ordering alone.

**Revised interpretation**: Temporal ordering is the *dominant* requirement (not sequential processing per se). Sinusoidal PE tells cross-attention WHEN each frame occurred; this is sufficient to capture most of the gait dynamics. The GRU's additional gain comes from its causal structure — each hidden state is a running summary of all prior frames, accumulating orientation history more efficiently than attention over all frames simultaneously. Both matter, but the ordering/temporal-context distinction is the key axis.

**Mechanistic insight (H1_v3 null result)**: Using 30fps AMASS pseudo-GT future poses (99% future step coverage) gives best minADE=0.6567 — virtually identical to H1_v2 (0.6532, zero future poses). **The trajectory benefit from pose conditioning comes entirely from the past pose GRU encoder, not from the quality of future pose supervision.** Better future GT for the pose decoder does not translate into better trajectory prediction.

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
- **Pose improvement is robust to loss weight (full model)**: All weights in [0.05, 0.2] give 2.8–3.2% improvement with all losses combined. Weight=0.1 is optimal but [0.05, 0.2] is a stable operating regime for the full model.
- **Geo_only weight has a SHARP OPTIMUM at w=0.1**: w=0.05→0.6786 (−1.5%), w=0.1→0.6231 (−7.6%), w=0.2→0.6660 (−1.3%). Unlike the full model's broad plateau, geo_only has a narrow peak. Both directions from w=0.1 collapse to near-baseline performance. This reveals a delicate balance: the geodesic loss must be strong enough to shape GRU orientation features but not so strong it dominates the trajectory objective.
- **Source of improvement localized to past pose encoding (H1_v3 null result)**: Using 30fps AMASS pseudo-GT future poses (99% future step coverage) gives minADE=0.6567, nearly identical to H1_v2 (0.6532) which had zero future pose GT. The trajectory benefit comes entirely from the GRU encoder processing past poses, not from supervising the pose decoder with better GT. This clarifies the mechanism: pose encoder → cross-attention with trajectory decoder → better trajectory queries.
- **Geodesic loss is the dominant supervision signal (H3 ablation)**: geo_only achieves best minADE=0.6231 (−7.6% vs baseline), outperforming the full model (0.6532, −3.2%). This shows MPJPE and cls_pose losses actively hurt when combined with geo. The rotation-space loss in SO(3) provides richer gradient signal to the GRU encoder than joint-space L1, likely because rotations encode orientation/gait information more directly than joint positions. gmm_only (0.6467, −4.1%) also outperforms mpjpe_only (0.6576, −2.5%), suggesting the WTA regression structure is more informative than point-wise L1 on joints.
- **Temporal ordering is critical; sequential integration adds incrementally (H5 + H5b)**:
  - Cross-attn, no PE (H5): 0.6765 — bag-of-items fails, ≈ baseline.
  - Cross-attn + sinusoidal PE (H5b): 0.6337 — PE recovers 79% of GRU's advantage.
  - GRU geo_only: 0.6231 — sequential integration provides a further 1.6% gain.
  Temporal ordering is the dominant requirement for geodesic supervision to produce trajectory-useful features. GRU's causal accumulation (running hidden state) captures slightly more than PE alone.
- **HD map dominates pose signal (H6)**: Adding HD map polylines reduces no-pose minADE by 27.7% (0.6745→0.4880) and pose minADE by 23.0% (0.6231→0.4797). Within-map, pose still improves by 1.7% (0.4880→0.4797), confirming a small but real complementary contribution. The shrinkage from 7.6% to 1.7% is the key quantity: body orientation and map routing are partially redundant representations of agent heading intent. Without map, pose compensates for missing spatial routing context; with map, that gap largely disappears.
- **Full-Waymo pretraining is the largest single lever (H7→H8/H9)**: H7 pretrains the trajectory-only backbone on 231k pedestrian agent-batch examples across 487k Waymo scenes (30 epochs, USE_POSE_ENCODER=False). Two fine-tune variants on the 579-scene subset: **H9** (no pose encoder, control) and **H8** (pose encoder + residual zero-init `pose_fuser`). Results: H9 minADE=0.3861 / minFDE=0.8166; H8 minADE=0.3749 / minFDE=0.7769. Compared to from-scratch H6 baselines (0.4880 no-pose, 0.4797 with-pose), pretraining yields a consistent ~21% improvement *independent* of pose, and pose adds an additional 2.9% on top of the H7 backbone (vs only 1.7% from scratch). The dataset-size bottleneck (579 train scenes) was the dominant limiter; pretraining on 200× more data yields a larger gain than any architectural ablation in this work, and pose's contribution is more visible against a strong trajectory prior than against noisy from-scratch features.
- **Convergence dynamics differ**: H1_v2 reaches best minADE earlier in training and has more variance across epochs (likely from noisy pose losses). H2_v2 plateaus more smoothly. Both settle around 0.67-0.70 after LR decay.
- **Pose loss weights matter critically for stability**: At weight=1.0, training diverges to NaN at epoch 3 due to gradient explosion through SMPL. At weight=0.1, training is stable for 30 epochs.
- **Small dataset limits absolute performance**: minADE ~0.65 is far from SOTA (~0.3 on full Waymo). With 579 training scenes vs 486k in full MTR, the gap is expected. The relative H1 vs H2 comparison is still valid.

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
5. **[ANSWERED] The 3.2% gain is consistent at minADE and minFDE, and robust across loss weights [0.05, 0.2].** Unlikely to be training noise given consistent direction across all ablations.
6. **[ANSWERED] Optimal pose loss weight is 0.1**, but [0.05, 0.2] is a stable operating regime (all within 0.4%).
7. **[ANSWERED] Does better future pose GT improve trajectory prediction?** NO — H1_v3 (30fps AMASS, 99% future coverage) gives minADE=0.6567 vs H1_v2 (zero future GT) minADE=0.6532. Future pose supervision quality does not drive trajectory improvement; the GRU past encoder is the mechanism.
8. **[ANSWERED] Geodesic loss alone is the dominant driver.** geo_only (0.6231, −7.6%) outperforms the full model (0.6532, −3.2%). MPJPE and cls_pose losses hurt when combined with geo.
9. **[ANSWERED] Geo weight for geo_only has a sharp optimum at w=0.1**: w=0.05→0.6786 (−1.5%), w=0.1→0.6231 (−7.6%, best), w=0.2→0.6660 (−1.3%). Both sides of the optimum give dramatically worse results. The geodesic signal needs exactly the right balance — too weak (0.05): GRU ignores rotation supervision; too strong (0.2): dominates trajectory objective causing gradient conflict with cls/reg/vel.
10. **[ANSWERED] Temporal ordering is the dominant requirement; GRU adds incrementally.** H5b (cross-attn + sinusoidal PE) achieves 0.6337 — recovering 79% of GRU's advantage over baseline. GRU geo_only (0.6231) adds a further 1.6% via causal sequential integration. Both temporal ordering and sequential processing contribute; ordering dominates.
11. Is there a way to evaluate pose prediction quality separately from trajectory quality?
12. **[ANSWERED] Does HD map context change the pose benefit?** YES — with map, pose benefit shrinks from 7.6% to 1.7% (0.4880→0.4797). Map and pose partially share information about agent heading. Both still help, but the interaction is subadditive: map+pose is not 7.6%+27.7% better than baseline, it is only ~29% better total.
13. **[ANSWERED] Was the small-dataset (579 scenes) the binding constraint?** YES — pretraining the backbone on the full Waymo set (231k pedestrian examples) and fine-tuning on the 579-scene subset gives minADE 0.3749 vs 0.4797 from-scratch (−21.9%). Single largest lever in this work. Architectural choices interact with data scale: ablations on 579 scenes underestimate the value of pretrained representations.
14. **[ANSWERED] Does pose still help on the pretrained backbone (controlling for pretrain effect)?** YES, and slightly more than from scratch — 2.9% (0.3861 H9 → 0.3749 H8) vs 1.7% (0.4880 → 0.4797 H6). The 21% pretrain gain is independent of pose; pose adds ~3% on top. Pose's contribution is more visible against a strong trajectory prior than against noisy from-scratch features.

## Optimization Trajectory

| Run | Hypothesis | Pose Weight | Best minADE | Δ vs baseline | Notes |
|-----|-----------|-------------|-------------|---------------|-------|
| run_002 | H2 (no pose) | 0.0 | 0.7724 | — (baseline) | ⚠ INVALID — 8 val samples |
| run_001 | H1 (full pose) | 1.0 | 0.6114 | −21% | ⚠ INVALID — 8 val samples |
| **run_004** | **H2_v2 (no pose, real GT)** | 0.0 | **0.6745** | — (new baseline) | ✓ VALID |
| run_005 | H1_w05 (pose, real GT) | 0.05 | 0.6557 | −2.8% | ✓ VALID |
| **run_003** | **H1_v2 (pose, real GT)** | **0.1** | **0.6532** | **−3.2%** | ✓ VALID ← optimal |
| run_006 | H1_w2 (pose, real GT) | 0.2 | 0.6542 | −3.0% | ✓ VALID |
| run_007 | H1_v3 (30fps AMASS future) | 0.1 | 0.6567 | −2.6% | ✓ VALID — null result (future pose GT irrelevant) |
| run_008 | H3 mpjpe_only | 0.1 | 0.6576 | −2.5% | ✓ VALID |
| run_009 | H3 gmm_only | 0.1 | 0.6467 | −4.1% | ✓ VALID |
| **run_010** | **H3 geo_only** | **0.1** | **0.6231** | **−7.6%** | ✓ VALID ← NEW BEST |
| run_011 | H3 geo_w=0.05 | 0.05 | 0.6786 | −1.5% | ✓ VALID — geo signal too weak |
| run_012 | H3 geo_w=0.2 | 0.2 | 0.6660 | −1.3% | ✓ VALID — too strong, dominates trajectory |
| run_013 | H5 cross-attn | 0.1 | 0.6765 | +0.3% | ✓ VALID — no improvement (≈ baseline) |
| run_014 | H5b cross-attn+PE | 0.1 | 0.6337 | −6.0% | ✓ VALID — PE recovers 79% of GRU advantage |
| run_015 | H6 no_pose+map | 0.0 | 0.4880 | −27.7% | ✓ VALID — HD map dominant signal |
| run_016 | H6 geo_only+map | 0.1 | 0.4797 | −28.9% | ✓ VALID — best from-scratch |
| run_017 | H7 full-Waymo pretrain | 0.0 | — | — | ✓ VALID — pretrain backbone (no eval) |
| run_019 | H9 no_pose+map, H7 pretrain → finetune | 0.0 | 0.3861 | −42.8% | ✓ VALID — H7 ablation, no pose |
| **run_018** | **H8 geo+map, H7 pretrain → finetune** | 0.1 | **0.3749** | **−44.4%** | ✓ VALID ← NEW OVERALL BEST |

## CRITICAL CORRECTION (2026-06-12): Geodesic loss is INERT in the 10fps condition

**Discovery**: In the main (10fps) data condition, future pose GT is all-zero for
99.99% of steps (verified: 5 of 91,520 future pose-target steps nonzero across the
entire training split — boundary rounding artifacts). A zero 6D target maps to the
all-zero 3×3 matrix under Gram–Schmidt, so tr(R_pred^T R_gt) ≡ 0 regardless of the
prediction → geodesic loss = arccos(−1/2) ≈ 2.0944 constant, with **identically zero
gradient** (verified numerically: `pred.grad == 0` exactly, while a random-GT control
gives nonzero grads).

**Consequences**:
1. Run_010 ("geo_only" = geo 0.1 + gmm 0.1) and run_009 ("gmm_only" = gmm 0.1) were
   functionally THE SAME experiment (geo term contributed a constant, no gradient).
   Their gap (0.6231 vs 0.6467) is pure run-to-run noise → noise scale ≈ ±0.02.
2. The paper's central claim "geodesic supervision is the critical ingredient
   (−7.6%)" was an artifact of comparing two replicates of one config.
3. The "geodesic weight sweep" (runs 011/012) actually swept the WTA-L1 weight
   (geo inert) → relabel as gmm/WTA-L1 weight sensitivity.
4. The `nll_loss_pose_gmm` loss is NOT a GMM NLL — it is a winner-takes-all masked
   L1 regression on 6D rotations (no sigmas, no likelihood). Paper mislabeled it.
5. All "geo+gmm" cells (H6 map runs, H8 finetune) are functionally "WTA-L1 only" —
   their pose-vs-no-pose comparisons remain valid, but supervision labels change.
6. H1_v3 (30fps) is the only condition where geodesic/MPJPE got real gradients;
   its null result (0.6567 ≈ 0.6532) stands and now means: even REAL future-pose
   supervision does not help trajectory accuracy.

**Other corrections found in the same audit**:
- Validation split is 5,171 scenes / 8,282 pedestrians (6,688 = 80.8% evaluated),
  NOT "97 validation scenes" as the paper claimed. Training: 579 scenes / 1,348
  pedestrians (1,144 usable prediction targets).
- GRU is 2-layer (cfg NUM_LAYER_IN_POSE_GRU=2), not single-layer; encoder has 6
  attention layers, not 4; optimizer is AdamW lr=1e-4 with step decay (not cosine
  1e-3); dropout 0.1 (not none). Paper appendix corrected.
- best_model.pth tracking is broken (tracks mAP which is hardcoded 0.0 → best_model
  is always epoch 1). Paper numbers come from per-epoch eval logs, which are fine.
- Pose provenance: poses are estimated from Waymo LiDAR by Waymo-3DSkelMo
  (LiDAR-HMR per-frame SMPL + HuMoR/NeMF motion-prior refinement), NOT
  "AMASS-derived". The pipeline snaps root orientation to past-trajectory heading
  when the raw estimate deviates >90° while moving → root-orientation channel
  partially encodes past heading by construction (no future leak; eval is sound).

**Remediation (2026-06-12)**: 42-run multi-seed matrix completed (12 cells × 3
seeds 101/202/303 + seeds 404/505 for headline cells; identical protocol).
Per-pedestrian metrics saved per epoch (metrics_epoch_N.pkl) for paired
bootstrap CIs over 6,688 evaluated validation pedestrians. Analysis:
tools/scripts/analyze_multiseed.py → experiments/multiseed_analysis.json.

### Multi-seed results (best-checkpoint minADE, mean±std over seeds)

| cell        | config                          | minADE          | per-seed |
|-------------|---------------------------------|-----------------|----------|
| baseline    | no pose (5 seeds)               | 0.6604 ± 0.0143 | .655 .683 .664 .654 .646 |
| geo_pure    | GRU, no active aux (5)          | 0.6545 ± 0.0271 | .696 .665 .639 .626 .647 |
| wta01       | GRU, WTA-L1 (5)                 | 0.6853 ± 0.0651 | .639 **.785** .629 **.717** .657 |
| gmm_only    | GRU, WTA-L1 — same config (3)   | 0.7403 ± 0.1354 | **.897** .662 .662 |
| mpjpe       | GRU, MPJPE+WTA (3)              | 0.6769 ± 0.0421 | .724 .662 .644 |
| full        | GRU, all aux (3)                | 0.6574 ± 0.0237 | .668 .674 .630 |
| xattn       | XAttn no PE (3)                 | 0.6667 ± 0.0407 | .714 .641 .646 |
| xattn_pe    | XAttn + sinusoidal PE (3)       | **0.6371 ± 0.0026** | .640 .634 .637 |
| map_nopose  | no pose + map (3)               | 0.4876 ± 0.0033 | |
| map_wta01   | GRU WTA + map (3)               | 0.4854 ± 0.0057 | |
| ft_nopose   | no pose + map + pretrain (3)    | 0.3791 ± 0.0033 | |
| ft_wta01    | GRU WTA + map + pretrain (3)    | 0.3718 ± 0.0059 | |

### Key paired-bootstrap results (per-pedestrian, seed-averaged, n=6,688)

- baseline→xattn_pe: **−3.53% [−4.04, −3.04] p<1e-4** — the stable pose win.
- baseline→wta01 (GRU): **+3.78% WORSE** [+3.2, +4.3] — original headline config.
- baseline→geo_pure (no-aux): −0.89% [−1.4, −0.4] p=2e-4 — only GRU cell that helps.
- baseline→xattn (no PE): +0.96% worse — ordering is necessary.
- map_nopose→map_wta01: −0.45% [−0.90, −0.01] p=0.046 — marginal.
- ft_nopose→ft_wta01: **−1.90% [−2.8, −1.0] p<1e-4** (pedestrian bootstrap) — pose's best realistic case.
- baseline→map_nopose: −26.2%; map→ft (pretrain): −22.3% — context dominates.

### Hierarchical bootstrap (2026-07-08): which conclusions survive resampling SEEDS

The paired bootstrap above marginalizes over seeds (each agent = its seed-mean), so
its CIs are pedestrian-sampling only. `analyze_multiseed.py` now also runs a
**hierarchical** bootstrap (resample seeds, then pedestrians). It is the conservative
test the paper now relies on for significance:

- **Survive** (p≤0.01): baseline→xattn_pe **−3.5%** [−4.8,−1.6] p<1e-4; xattn→xattn_pe
  −4.5% p=0.008; baseline→map_nopose −27% and map/pretrain −22% p<1e-4.
- **Lose significance**: baseline→geo_pure −0.9% → **p=0.59**; map_nopose→map_wta01
  −0.45% → **p=0.62**.
- **Weakens to marginal**: ft_nopose→ft_wta01 −1.9% → **p=0.10** (CI just crosses 0);
  only 3 seeds. The −1.9% "robust" claim is downgraded to marginal in the paper.

Net: only the ENCODER effect (xattn+PE) and the context effects (map, pretrain) are
seed-level significant. All small pose margins (≤2%) are not, at three seeds.

### Phase 4 (2026-07-08): xattn+PE in realistic conditions + root-orientation ablation

Six new cells × 3 seeds (101/202/303), same protocol; sourced in
experiments/multiseed_analysis.json. minADE mean±std:
- **map_xattn_pe 0.4792±0.0021** (xattn+PE + map, from scratch)
- **ft_xattn_pe 0.3733±0.0038** (xattn+PE + map + H7 pretrain finetune)
- **norootorient 0.6633±0.0031** (xattn+PE, root-orientation channel zeroed)

Key new pairs (paired / hierarchical):
- map_nopose→map_xattn_pe: **−1.73%** [hier CI −1.4,−0.3%] **p=0.004** — the stable
  encoder DOES give a seed-significant pose benefit under map, ~4× the GRU cell's
  −0.45% (which was p=0.62). Closes the paper's biggest hole: xattn+PE was never
  run with map; it helps, and significantly.
- ft_nopose→ft_xattn_pe: −1.51% [hier p=0.063] — marginal, like GRU ft (−1.9%).
- ft_wta01→ft_xattn_pe: +0.40% p=0.77 — encoder choice does NOT matter once the
  backbone is pretrained (xattn+PE ≈ GRU at pretrained scale).
- **baseline→norootorient: ~0 (p=0.51); xattn_pe→norootorient: +4.11% p<1e-4** —
  zeroing the root-orientation channel ELIMINATES the entire −3.5% xattn+PE pose
  benefit (model reverts to ≈baseline). The pose gain is essentially all in the
  root-orientation channel, which the Waymo-3DSkelMo pipeline partly derives from
  past-trajectory heading (orientation-snapping). This quantifies the circularity
  the provenance caveat flagged: much of the "pose benefit" is past heading
  re-entering through a side channel. Now integrated into the paper (§4.4 root-
  orientation paragraph, xattn+PE rows in the map/pretrain tables, §4.5 rewrite,
  Limitations + intro/conclusion caveats); paper still fits 9 main pages.

### Final corrected story

1. Whether pose helps is decided by the ENCODER, not the aux loss: xattn+PE gives a
   stable −3.5%; the GRU is heavy-tailed (5 of 25 no-map pose runs ≥0.71, worst
   0.897 = +36%) and on average WORSE than baseline. The 8 pooled replicates of the
   "geo_only"/"gmm_only" config span 0.629–0.897 (0.7060±0.0921).
2. Temporal ordering carries the signal (bag-of-frames useless; +PE best-in-class).
   The draft's "GRU beats attention by 1.6%" REVERSES: GRU trails xattn+PE by +7.6%.
3. Map context (−26%) and pretraining (−22%) dominate pose, stabilize training
   (zero failures in 12 map-enabled seeds), and shrink pose to −0.45%/−1.9%.
4. Pose benefit grows with backbone quality: −0.45% (scratch+map) → −1.9%
   (pretrained), adjacent non-overlapping CIs.
5. Paper rewritten around the audit + seed-replicated measurement
   (new title: "How Much Does Body Pose Help Pedestrian Trajectory Forecasting?
   A Seed-Replicated Study and Auxiliary-Loss Audit on Waymo").

## Paper Status (2026-04-03 — CONCLUDE)

Paper at `paper/iclr2026/main.tex` compiles cleanly to 11 pages (8 main + 1 references + 2 appendix). 15 verified citations. All numbers consistent across tex/yaml/html.

**Related Work §2.3 correction (2026-04-03)**: Literature search found three prior pose+trajectory papers that our original "no existing work" claim incorrectly ignored:
- salzmann2023robots (IEEE RA-L 2023): 3D skeletal keypoints, robot navigation
- saadatnejad2024socialtransmotion (ICLR 2024): 2D/3D skeleton pose, pedestrian benchmarks
- gao2025socialpose (IEEE T-ITS 2025): skeleton body language, social navigation

Our contribution remains distinct: parametric SMPL 6D rotations (not skeleton keypoints), Waymo+MTR framework, first systematic geodesic vs. MPJPE vs. GMM supervision ablation, temporal encoding (GRU vs cross-attn) decomposition.

**Remaining human tasks before submission:**
1. Author names/affiliations (uncomment `\iclrfinalcopy`)
2. ~~ICLR 2026 LLM disclosure~~ — completed: added to main.tex §Acknowledgements
3. Multi-seed runs for confidence intervals (recommended)

**2026-04-07 polish**: Fixed introduction structure — "three principal findings" was immediately followed by "A fourth finding clarifies the mechanism", creating an inconsistency. Moved future-pose null result into a 4th bullet and updated "three" → "four principal findings". Recompiled: 11 pages, no errors.
