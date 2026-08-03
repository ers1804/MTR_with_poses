# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-01 | bootstrap | Explored codebase. MTR extended with SMPL pose: GRU encoder, pose heads, MPJPE+geodesic+NLL losses. Data: 579 training scenes at final_10fps/training/. Found critical timestamp bug (µs vs s). Formed 5 hypotheses. Fix config DATA_ROOT path. |
| 2 | 2026-04-01 | bootstrap | Fixed timestamp unit bug in waymo_pose_dataset.py (divide by 1e6). Updated mtr+pose_data.yaml DATA_ROOT to actual data path. Set up autoresearch workspace. Set up /loop 20m for agent continuity. |
| 3 | 2026-04-01 | bugfix | Fixed SMPL betas mismatch: SMPL_NEUTRAL.pkl has 300-dim shapedirs but decoder passes 10-dim betas. Fix: truncate th_shapedirs[:,:,:10] and th_betas[:,:10] in mtr_decoder.py after SMPL_Layer init. |
| 4 | 2026-04-01 | bugfix | Fixed dataset IndexError: get_interested_agents fallback selected future-only agents (no valid past), causing valid_past_mask=False → track_index=-1. Fix: fallback now requires past-valid agents. |
| 5 | 2026-04-01 | bugfix | Fixed BatchNorm error on batch-size-1 (last batch). Fix: DATALOADER_DROP_LAST=True in config. Added __getitem__ retry on ValueError/IndexError for robustness. |
| 6 | 2026-04-01 | experiment | Launched H1 training run (H1_pose_all_losses): 30 epochs, batch=10, full pose model. Pipeline verified end-to-end. ~80s/epoch → ~40min total. |
- 2026-06-12: AUDIT — geodesic loss proven inert (zero grad vs zero future-pose GT);
  "geo_only"≡"gmm_only" → headline −7.6% was seed noise. Ran 42-run multi-seed
  matrix (12 cells, seeds 101-505): xattn+PE is the stable pose win (−3.5% CI
  [−4.0,−3.0]); GRU heavy-tailed (mean worse than baseline); map −26.2%; pretrain
  −22.3%; pose on pretrained −1.9% [−2.8,−1.0]. Paper fully rewritten
  ("How Much Does Body Pose Help...? A Seed-Replicated Study and Auxiliary-Loss
  Audit on Waymo"). Figures pending regeneration.
  [SUPERSEDED 2026-07-21 — the −3.5% and −1.9% claims below did not survive n=5.]
- 2026-07-08: HYGIENE + CORRECTNESS (action plan P0/P1). Tracked the 8 untracked
  experiment configs (a fresh clone could not run a single headline cell); untracked
  LaTeX build artifacts. Root-caused the same bug class as the geodesic audit: NO
  future-pose validity mask anywhere → all four pose losses trained against all-zero
  targets (MPJPE was a full-gradient T-pose regularizer). Added center_gt_poses_mask
  (pose-row-nonzero ∧ traj-valid) + past-pose mask fix, mAP-placeholder removal
  (best_model was frozen at epoch 1), pose-weight defaults 0.0, grid-mapping
  tolerance, cls gating. TDD: test/test_pose_masks.py written failing first, then
  green; 2-epoch smoke run clean.
- 2026-07-08: STATISTICS — added a hierarchical bootstrap (resample seeds, then
  pedestrians) alongside the pedestrian-only paired bootstrap. Under it, geo_pure
  (−0.9%) and map-pose (−0.45%) lose significance and ft-pose (−1.9%) weakens to
  marginal; only the encoder and context effects survive. Paper significance
  language rebuilt on the conservative test; main text cut to 9 pages.
- 2026-07-08/13: PHASE 4 — xattn+PE run in the realistic conditions it had never
  been run in (map, pretrain) + root-orientation ablation. map_xattn_pe −1.7%
  (p=0.004); GRU+map ≈ nothing; ft-pose collapses once ft_nopose reaches n=5.
- 2026-07-13: HEADLINE #2 RETRACTED — xattn_pe seeds 404/505 (one a heavy-tail
  failure) take the cell to 0.6549±0.0278: the "stable −3.5%" was itself a 3-seed
  artifact. Masked audit re-run: masking removes the WTA catastrophic tail but adds
  no signal; masked baseline unchanged (p=0.95).
- 2026-07-15: 30 fps future-pose supervision seed-replicated (the one condition with
  real geodesic/MPJPE gradients): null holds, −1.4% p=0.13. Fixed a P1.6 regression
  (uniform-spacing assertion wrongly rejected valid 30 fps files with dropped frames).
- 2026-07-20: FULL PROJECT REVIEW (to_human/project_review_2026-07-20.md). Fixed
  stale 3-seed figures that still supported the retracted claim, a false "narrowest
  band" statement, a reproducibility overclaim ("mask-insensitive"), and added a
  multiplicity note. Found a CODE-VERSION MIXING FLAW: the P1.1 mask fix changed the
  training distribution of every aux-active cell, so the 5-seed xattn_pe cell mixed
  3 pre-fix + 2 post-fix runs. Repaired via a git worktree pinned to the pre-fix
  commit; mixed runs preserved under their own cell.
- 2026-07-21: FINAL PICTURE (34-run queue, every cited cell now n≥5 with clean
  provenance). (a) The xattn_pe collapse REPLICATES on clean pre-fix code
  (0.6648±0.0381, +0.7% vs baseline, p=0.83) — retraction confirmed; the
  PE-"ordering" mechanism dissolves with it (p=0.76) → no encoder property is
  resolvable in the no-map regime. (b) MECHANISM REVERSED: the surviving
  map×xattn+PE benefit (−1.57%, p<1e-4) is NOT heading re-entry — it survives
  zeroing the heading-coupled root channel (0.4800, p=0.58) and an agent-centric
  pose frame (0.4801, p=0.65) → genuine body-configuration signal. (c) Masked-audit
  "+5.5% still hurts" dead at n=5; pretrain/30 fps nulls firm; 26% of no-map pose
  runs fail. (d) MR@2m added as external anchor (0.239/0.145/0.093); official Waymo
  mAP dropped (predictions not retained) and disclosed. Paper, findings.md,
  research-state.yaml and the slide deck all rewritten to this.
