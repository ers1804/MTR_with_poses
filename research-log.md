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

## 2026-08-18 — CRITICAL: original pose export had cross-scene id-collision bug; all MTR pose data was effectively wrong-pose

While preparing the WOMD-scale re-export (Erik's question "didn't the pipeline run on all of WOMD?"), traced the full 3DSkelMo funnel:
extract 199,784+ train scenes → NeMF generation 1,876,000 clips (~326k scenes, 12 groups) → clips/ per-scene staging 299,222 train + 27,156 val → final_10fps **579 train / 5,171 val**.

The collapse is NOT truncation. `export_final_data.py` built `target_files` keyed by object id alone (`file.rsplit('-',1)[0]`) via os.walk over ALL scenes and BOTH splits. WOMD object ids are scene-local (25.3% of ids in a 201-scene sample appear in >1 scene; ids range ~56–10,467). Every output npz therefore contains the body motion of the FIRST same-id pedestrian in walk order and the metadata (timestamps, floor, offset, betas) of the LAST; one output per unique id (~5.7k scene dirs).

Smoking gun: final_10fps/training/11f5bcd637d947ce/8899.npz pose_body rows are bit-identical (0.0000) to clips content of scene 903a915cc3ddf5a object 8899, walked ~275k scenes earlier; own-scene content differs by 0.137 rad. 19/20 sampled objects mismatch their own scene (mean 0.127 rad vs sequence std 0.324); the 1 exact match is a non-colliding id — which also validates the per-scene reimplementation bit-for-bit.

Consequence: MTR configs read final_10fps directly (DATA_ROOT), so every pose-conditioned run in the project trained and evaluated on scrambled pedestrian identity — the "correct pose" condition never existed. This offers a data-level explanation for the capacity ladder (correct ≈ wrong ≈ constant) and requires re-testing the central claims on clean data. 7th and largest self-caught artifact.

Fix + action: `Waymo-3DSkelMo/scripts/export_final_data_fixed.py` (per-scene grouping, resumable, 24 workers ≈ 14 scenes/s). Full re-export of ~326k scenes (~6.5 h) launched to /home/erik/big1/waymo_motion_poses/final_10fps. Next: rebuild the pose subset from clean data, re-run core cells (baseline vs pose vs wrong-pose with map, n=5), then decide paper reframe. nuScenes/AV2 extension assessed and rejected (nuScenes: median 3 LiDAR pts/ped; AV2: no sensor data on disk, no LiDAR in forecasting split; WOMD camera = VQ tokens only, so image-HMR impossible on WOMD).

## 2026-08-19 (later) — clean-data core triad at n=5: the n=3 harm signal dissolves; full null

Extension seeds 404/505 complete (pre-registered trigger: hier p<0.05 at n=3). Final n=5 picture on clean data
(579-scene subset, 6,348 train peds):
- map_nopose_fix      0.4599 ± 0.0206  (seed 404 is a 0.4954 outlier — the n=3 mean was flattered)
- map_xattn_pe_fix    0.4620 ± 0.0094  → vs no-pose +0.45%, hier p=0.76 (was +1.85%, p=0.037 at n=3)
- map_shufflepose_fix 0.4589 ± 0.0075  → vs no-pose −0.21%, hier p=0.96
- correct vs shuffled: −0.66%, hier p=0.46
- map_wta01_fix (n=3) +1.44% vs no-pose, hier p=0.33 — aux loss on real targets: no effect

VERDICT: with genuinely correct poses (proxy-validated), 4.7× more pedestrians than the corrupt era, and the
aux-loss recheck, the from-scratch map regime shows a complete null: no help, no harm, content
indistinguishable from scrambled content. The n=3 harm report is the project's 8th self-caught
seed artifact. Remaining levers: finetune regime (running), 299k-scene scale-up, better pose quality (image-HMR).

## 2026-08-20 — finetune triad (clean data, n=3): SHUFFLED poses significantly best; n=5 extension launched

- ft_nopose_fix      0.3683 ± 0.0033
- ft_xattn_pe_fix    0.3690 ± 0.0045  → vs no-pose +0.19%, hier p=0.80 (null)
- ft_shufflepose_fix 0.3636 ± 0.0009  → vs no-pose −1.27% hier p=0.016; vs correct −1.46% hier p=0.037

The wrong-pose control BEATS both no-pose and correct-pose in the finetune regime, with unusually tight seeds.
If it survives n=5 (seeds 404/505 queued: run_fixdata_ft_extend.sh), the reading is structured-noise
regularization during finetuning — realistic-but-uninformative input helps where correlated input does not.
Given 8 prior 3-seed artifacts, no interpretation before n=5.

## 2026-08-20 (later) — finetune triad at n=5: shuffle advantage dissolves; project-wide clean-data null is final

- ft_nopose_fix      0.3692 ± 0.0031
- ft_xattn_pe_fix    0.3708 ± 0.0040  → vs no-pose +0.44%, hier p=0.49
- ft_shufflepose_fix 0.3676 ± 0.0055  → vs no-pose −0.44%, hier p=0.46 (was −1.27% p=0.016 at n=3 — 9th artifact)
- shuffle vs correct: −0.88%, hier p=0.11

COMPLETE CLEAN-DATA PICTURE (all hierarchical, 579-scene subset):
from-scratch triad (n=5): null | aux pose loss (n=3): null | finetune triad (n=5): null.
Pose conditioning contributes nothing detectable in any regime on clean data — no help, no harm,
correct indistinguishable from scrambled. Remaining open levers: pose-quality ceiling
(image-HMR / full-pipeline MPJPE on Perception) and scale (299k clean scenes).

## 2026-08-20 (evening) — MPJPE protocol bug (artifact #10): 160.6mm was inflated; corrected 87.6mm; image-HMR does not beat LiDAR

Image-HMR validation (HMR2.0/4D-Humans, same 347 frames / 113 peds / 13 joints / neutral SMPL as the LiDAR eval)
surfaced a bug in OUR harness: root_aligned_mpjpe.py indexed the 30fps joint sequence by track-entry index, but
Waymo frame i sits at fps30 index 1+3i (verified on raw timestamps; buggy protocol reproduces 160.6mm exactly).

Corrected numbers (timestamp-aligned):
- LiDAR-HMR: 87.6mm root-aligned (median 75.0) / 72.4mm PA-no-scale   [was published as 160.6]
- image HMR2.0: 106.2mm root-aligned / 82.5mm PA — LOSES to corrected LiDAR-HMR by ~18mm; error flat vs
  crop height (126-665px). RGB-based estimation on Waymo cameras is NOT the better pose source.
- Per-joint (LiDAR ts-aligned): ankles ~104-107, wrists ~128-136 — well below stride amplitude (300-600mm),
  so the paper's "gait at the noise floor" argument is retracted.

Story impact: pose quality can no longer explain the null. Clean data + decent poses + all-null (n=5, both
regimes) = the null is about information redundancy at this scale, not data quality. Data:
experiments/image_hmr_mpjpe.json; code /home/erik/big1/image_hmr/eval/. 10th self-caught artifact.

## 2026-08-20 (night) — full-pipeline MPJPE: 92.7mm; HuMoR/NeMF adds ~5mm, concentrated in gait joints; quality chain complete

Full Perception pipeline output (what the predictor consumes) vs GT keypoints, corrected ts-aligned protocol,
same 347 frames / 113 peds, zero attrition:
- LiDAR-HMR stage 87.6mm root-aligned / 72.4 PA  →  full pipeline 92.7 / 77.8 (paired +5.0mm, p=3e-4)
- Damage concentrated in legs: knees +19mm (73→92), ankles +17/+25mm (→121/131) — NeMF inbetweening washes
  out gait phase; upper body neutral. Betas vs zero-betas: immaterial.
- Also verified: the export id-collision bug is HARMLESS on Perception (22,648 ids, 0 cross-scene collisions —
  base64 globally-unique ids, unlike WOMD numeric ids). The MPJPE data was never affected by that bug.

FINAL QUALITY CHAIN: LiDAR-HMR 87.6 → +HuMoR/NeMF 92.7 (gait joints ~121-131) → all well below stride
amplitude. Alternative RGB source (HMR2) 106.2 — worse. Pose quality cannot carry the all-null; the mild
gait-phase smoothing cost is the only quality caveat worth a paper sentence. Experimental program of the
2026-08 audit cycle is COMPLETE. Data: experiments/full_pipeline_mpjpe.json.

## 2026-08-21 — scale-up launched: nested ladder on clean data

Join verified: all 299,220 clean pose scenes have processed_scenarios pkls (checked against the 486,995-scene
WOMD training index) — full-scale pose-conditioned training is unblocked; only compute binds.

Ladder (nested supersets, validation IDENTICAL to every clean-data run = 5,171 scenes / 20,034 peds):
  579 scenes / 6,348 peds (done, all-null)  →  5k / 31,302 (4.9x)  →  25k / 142,807 (22.5x)  →  [299k / ~1.8M]

Rung-1 sizing note: a first attempt at 10 epochs (decay [3,7,8]) UNDERTRAINED — best 0.4971 at epoch 5 vs
0.4599 for the 579-scene/30-epoch runs, curve still bouncing at 0.512 after early LR decay. Discarded and
relaunched with the SAME 30 epochs / decay [10,20,25] as the 579 runs, so data volume is the only difference.
Cost: ~3 h/run → ~27 h for the 9-run triad (nopose/xattn_pe/shufflepose x 3 seeds).

Implication for 25k: ~15 h/run x 9 = ~135 h locally — that rung, not this one, is the Helma decision point.
GPU is shared with Erik's navsim_eval (Latent_Diff) job; queue waits rather than preempting.

## 2026-08-21 (evening) — scale rung 1 halted for diagnosis: pose BRANCH degrades at 4.9x data; LR check running

Rung-1 (5k scenes, 30 epochs, matched schedule) first three runs, seed 101:
- nopose      0.4270  (vs 0.4584 at 579 scenes — scale HELPS the baseline by 6.8%)
- xattn_pe    0.5852  (+37% WORSE than nopose; eval curve degrades monotonically 0.586 -> 0.848)
- shufflepose 0.6775 at first eval (degrading too) -> it is the pose BRANCH, not pose content

Ruled out by direct measurement:
- pathological pose values (0 nonfinite, max|rot| 4.7 rad, identical to the 579 subset)
- pose/trajectory join failure (center-pose coverage 0.881 at 5k vs 0.891 at 579)
- eval-population mismatch (both 20,034 peds / 5,171 scenes), NaN (none), grad explosion (clip=1.0 active)
- AUX-LOSS HIJACK: instrumented loss components show geo/gmm_pose/cls_pose/mpjpe are ALL EXACTLY 0.0 --
  sequences are past-only (max 11 frames), so the future-pose mask is empty.

  IMPORTANT COROLLARY: the aux-loss cells (map_wta01*, geo/gmm variants) have NEVER exercised the pose
  losses on clean data -- they differ from the no-pose baseline only by having a pose ENCODER. The
  "auxiliary-loss audit" framing in the paper needs revisiting: on this data the losses are inert by
  construction (no future pose targets), not merely ineffective.

Remaining hypothesis: optimization/tuning -- at 5k each epoch is 4.9x longer, so the pose branch takes ~5x
more steps at LR 1e-4 before the first decay. Diagnostic launched (run_lrdiag.sh): nopose and xattn_pe both
at LR 5e-5, 8 epochs, seed 101. If the pose arm tracks the baseline at the lower LR, the "pose hurts at
scale" reading is a tuning artifact and the rung must be rerun with a schedule where BOTH arms converge.
Queue halted meanwhile to avoid burning ~18 h on a possibly mis-tuned grid.

## 2026-08-21 (late) — LR diagnostic: the 5k pose collapse is largely an OPTIMIZATION artifact

Both arms at LR 5e-5 / 8 epochs (seed 101), same data, same schedule:
- nopose  best 0.4690   pose best 0.5037  -> pose penalty +7.4%
vs at LR 1e-4 / 30 epochs:
- nopose  best 0.4270   pose best 0.5852  -> pose penalty +37%

Halving the LR shrinks the pose penalty 5-fold, so "pose conditioning collapses at scale" is NOT a property
of the data — the pose-conditioned model needs gentler optimization when each epoch is 4.9x longer (~5x more
steps at the initial LR before the first decay). Caveat: the 8-epoch schedule is too short — BOTH arms are
unconverged and degrade late (nopose 0.469 -> 0.7115), so the +7.4% is only indicative.

Consequence for the scale study: comparing architectures at a single LR tuned on the 579-scene subset is not
a fair test at 4.9x data. Each arm must be trained where IT converges (per-arm tuning, best-of-sweep), which
multiplies the run count: 3 cells x ~3 LRs x 3 seeds x 3 h = ~80 GPU-h PER RUNG. That is the real Helma
argument — not the raw data volume.

Launched now: rung-1 triad at LR 5e-5 / 30 epochs (proper length), seed 101, 3 runs ~9 h
(run_scale5k_lr5.sh). Combined with the existing 1e-4 numbers this gives a per-arm best-of-two comparison
before committing seeds.

## 2026-08-22 — SCALE RUNG 1 COMPLETE (5k scenes, 4.9x data, tuned LR, n=3): null persists

Tuned recipe (LR 5e-5, 30 epochs, decay [10,20,25]); validation identical to all clean-data runs:
- scale5k_lr5_nopose      0.4176 ± 0.0060
- scale5k_lr5_xattn_pe    0.4138 ± 0.0053   vs nopose  -0.90%, hier p=0.52
- scale5k_lr5_shufflepose 0.4181 ± 0.0068   vs nopose  +0.12%, hier p=0.85
- correct vs shuffled: -1.02%, hier p=0.44

Data DOES help the pipeline: nopose 0.4599 (579 scenes, n=5) -> 0.4176 (5k, n=3) = -9.2%. So the null is not
"model can't learn"; it is specific to pose conditioning. At 4.9x data the pose delta remains indistinguishable
from zero AND from its own wrong-pose control — same picture as 579 scenes, now with a genuinely stronger
baseline.

Optimization-sensitivity note for the paper (important methodological finding): the SAME comparison read
+37% (LR 1e-4, the recipe tuned on the 579 subset), +7.4% (LR 5e-5, 8 epochs), and -0.90% (LR 5e-5,
30 epochs, converged). Architecture comparisons at a new data scale are hostage to per-arm optimization
sensitivity; a single inherited LR manufactures large fake effects in BOTH directions.

Scale ladder so far (hier p, pose vs no-pose): 579 scenes +0.45% p=0.76 (n=5) | 5k scenes -0.90% p=0.52 (n=3).

## 2026-08-24 — Helma port prepared

Payload staged and verified on big1 (/home/erik/big1/helma_stage): the nested 25k training ladder
(25,000 scenes, 142,807 peds — contains the 5k rung and the original 579) plus the fixed 5,171-scene
validation split. 30 GB total: pose npz (1.7 GB) + processed_scenarios pkls (28 GB). Integrity checked
(40 random scene pairs load; 0 pose scenes missing a scenario pkl on either split).

Cluster facts (helma, NHR@FAU): partitions h100/h200 (24 h) and preempt (48 h); modules
python/3.12-conda + cuda/12.6.2 under gpu-env/2025; workspaces on /hnvme (Lustre, 1.2 P free).
IMPORTANT: $HOME is at 96 G/100 G and 483k/500k inodes -> conda env, conda/pip caches and
TORCH_EXTENSIONS_DIR all must live in the workspace (handled in slurm/00_setup_env.sh).

Scripts (slurm/, committed): 00_setup_env.sh builds the env and compiles MTR's CUDA extensions
(knn_cuda, attention_cuda) on a GPU node; 10_scale_sweep.sh runs a 27-task array
(3 cells x 3 LRs x 3 seeds, 12 concurrent) staging tars to node-local $TMPDIR; README.md carries the
porting gotchas. Transfer helpers live in the staging dir: pack_and_ship.sh, ship_code.sh (rsync of the
working tree, NOT a clone -- branch mtr_smpl has ~25 unpushed commits and the _scale5k_lr5 configs).

Design note: the sweep tunes LR PER ARM by construction. Comparing arms at one inherited LR is what
produced the +37% phantom "pose hurts" effect at the 5k rung.

Blocked on: workspace path (scripts default to /hnvme/workspace/v103fe12-mtr_pose) and go-ahead to
push 30 GB to the cluster.

## 2026-08-24 (cont.) — Helma port executed

Workspace: /hnvme/workspace/v103fe12-mtr_smpl. Shipped: repo working tree (2,118 files, rsync not clone --
branch mtr_smpl is 39 commits ahead of origin), poses_25k.tar (785 MB), scenarios_25k.tar (29 GB),
cluster_64_center_dict.pkl. Transfer ran at ~165 MB/s.

Cluster constraints that shaped the design:
- /hnvme file quota ~82k soft / 102k hard with 33k already used, while a torch conda env is 100k+ files
  -> env is built in node-local $TMPDIR and stored as ONE conda-pack archive; datasets stay as two tars;
  nothing is ever extracted into the workspace.
- $TMPDIR is wiped after each job -> every job re-stages env + data; outputs and the compiled .so files
  live in the workspace.
- Env builds only inside an allocation (salloc/srun), not on login nodes.

Three failures fixed en route (all cluster-porting issues, none scientific):
1. transient EACCES on a freshly created workspace (mount not settled) -> retry
2. `python setup.py develop` on modern setuptools routes through an isolated PEP 517 build with no torch
   -> ModuleNotFoundError; fixed with an install that skips build isolation
3. verification imported `knn_cuda` top-level; the extensions are `mtr.ops.knn.knn_cuda` and
   `mtr.ops.attention.attention_cuda` -> both .so files had in fact compiled fine

## 2026-08-24 (cont.) — Helma validation status

Env packed (3.0 GB single archive; workspace file count 35.6k of 82k -- the packed design cost ~2k files
instead of ~130k). Data verified staged on a compute node: 25,000 train + 5,171 val scenes.

Porting failures fixed and committed (all mechanical, none scientific):
4. conda-pack refuses envs containing editable installs -> pack around it (the editable path is the
   persistent workspace repo, so it still resolves after unpack)
5. `set -u` aborts on conda-pack's activate script ($CONDA_PREFIX unbound) -> relax strict mode across
   activation. This one killed a job in 44 s and would have killed all 27 identically.
6. smplpytorch is a TOP-LEVEL import in mtr_decoder.py, so training cannot import without it even though
   the SMPL layer only feeds the inert MPJPE loss; chumpy needed for it, installed without build isolation.
   Also SMPL_MODEL_DIR was a hardcoded workstation path -> overridden to the workspace copy at run time.
   (tensorflow / waymo_open_dataset are NOT needed: only data_preprocess.py and waymo_eval.py import them,
   neither on the training path; the workstation env lacks them too.)

Operational lesson (documented in slurm/README.md): polling the login node with a fresh ssh every ~60 s from
several watchers triggered connection refusal (kex_exchange_identification). Running jobs are unaffected but
visibility is lost. Use one multiplexed ssh (ControlMaster/ControlPersist) and poll at minute-plus intervals,
or have the job write status into $WS/logs and read that.

## 2026-08-24 (evening) — Helma sweep LAUNCHED

Validation task passed end-to-end on an H100 (first eval minADE 0.4970, a sane early value): env unpacked
from the packed archive, both data tars staged to node scratch (25,000 train / 5,171 val scenes), SMPL layer
constructed, compiled CUDA extensions working. Remaining 26 array tasks released (job 776244,
3 cells x 3 LRs x 3 seeds, 12 concurrent).

Porting fixes 7-9 (all committed):
7. chumpy 0.70 cannot import on numpy >= 1.24 (`from numpy import bool, ...`). The workstation env had a
   hand-applied one-line patch that existed nowhere in the repo and would have been lost on any fresh setup;
   the env build now applies it automatically and verifies the smplpytorch import before packing.
8. `module: command not found` — caused by my own ssh hardening: the wrapper opens a NON-login shell and
   `module` is a shell function from /etc/profile.d/zz-rrze-local.sh. Scripts now bootstrap the module system
   themselves, so they behave identically under sbatch / srun / ssh / interactive.
9. that same profile script references an unset $GROUP, so sourcing it under `set -u` aborted -> strict mode
   relaxed only around the sourcing.

SSH lockout prevention (Erik's request), now structural rather than procedural:
- slurm/ssh_config + slurm/helma_ssh.sh: multiplexed master connection reused 8 h (crucially covering the
  csnhr ProxyJump hop too — every `ssh helma` was previously TWO connections), new-master attempts
  rate-limited to one per 2 min behind a flock, progressive backoff after refusals.
- sweep jobs self-report START / DONE+best / EXIT rc into $WS/logs/status.txt, so the whole array is
  readable in ONE round trip instead of 27 Slurm queries. Watchers now poll at 15 min.

## 2026-08-26 — SCALE RUNG 2 COMPLETE (25k scenes, 22.5x data, Helma, per-arm LR sweep)

36 runs on Helma (3 cells x 4 LRs x 3 seeds), 0 failures. LR optimum is INTERIOR for every arm at
2.5e-5 (1e-4: 0.4329 / 5e-5: 0.3865 / 2.5e-5: 0.3711 / 1.25e-5: 0.3821 for no-pose), so the
extension resolved the boundary-optimum risk and per-arm best-of-sweep is trustworthy. All three
arms peak at the same LR -- no per-arm optimisation divergence at this scale.

Per-arm best (all at LR 2.5e-5, n=3):
- scale25k_nopose      0.3711 +- 0.0036
- scale25k_xattn_pe    0.3774 +- 0.0028   vs nopose +1.68%, hier p=0.0004
- scale25k_shufflepose 0.3770 +- 0.0064   vs nopose +1.58%, hier p=0.061
- correct vs shuffled: -0.09%, hier p=0.92   <-- content signal is nil

READING: at 22.5x data the pose BRANCH now carries a small but statistically resolvable COST
(+1.7%, p=0.0004), while pose CONTENT remains worth exactly nothing (correct vs shuffled 0.09%,
p=0.92). The cost is a capacity/optimisation tax that becomes measurable once the baseline is
strong enough and seeds tight enough to resolve it -- not evidence that poses carry harmful
information, since scrambled poses cost the same.

Scale ladder, pose vs no-pose (hierarchical): 579 scenes +0.45% p=0.76 (n=5) | 5k -0.90% p=0.52
(n=3) | 25k +1.68% p=0.0004 (n=3). Baseline keeps improving with data: 0.4599 -> 0.4176 -> 0.3711
(-19% overall), so the pipeline is far from saturated while the pose delta never turns positive.

CONCLUSION: the scale hypothesis is exhausted. More data makes the baseline better and pose
conditioning no better; the only thing that grows with scale is the cost of carrying the branch.

### Rung-2 final grid (all 36 runs, 0 failures)

  no_pose_with_map     2.5e-5=0.3711  1.25e-5=0.3821  5e-5=0.3865  1e-4=0.4329
  cross_attn_pe        2.5e-5=0.3774  5e-5=0.3787     1.25e-5=0.3801  1e-4=0.4224
  shufflepose          2.5e-5=0.3770  1.25e-5=0.3807  5e-5=0.3845  1e-4=0.4259

Every arm peaks at 2.5e-5 with worse values on BOTH sides -- interior optima, so the per-arm
best-of-sweep comparison is sound and the 1.25e-5 extension did its job. Note also how large the
tuning effect is compared with the effect under study: LR choice moves minADE by up to 17%, while
the entire pose content signal is 0.09%. Any single-LR comparison at a new data scale is therefore
dominated by tuning, not by the treatment -- the methodological point of this whole rung.

## 2026-08-26 — REVIEW: pose coordinate frame (a real, untested bug) + architecture integration

### 1. Coordinate frame: MISMATCHED by default

Verified in code and empirically on clean data:
- Trajectories ARE agent-centric: transform_trajs_to_center_coords translates by center_xyz, rotates
  positions/velocities by Rz(-center_heading) and subtracts center_heading from heading channels.
- Pose root orientation is left in WORLD frame. rotate_root_6d() exists but runs ONLY under
  AGENT_CENTRIC_POSE_ROT, and NO clean-data config (fixdata / scale5k / scale25k) enables it.
- Empirical confirmation (n=400 clean pedestrians): SMPL root yaw std 1.91 rad (not canonicalised),
  correlation with Waymo WORLD heading 0.72 (sin) / 0.72 (cos), median |difference| 13.6 deg.
  So the root channel encodes world-frame facing while every other feature for that agent is
  expressed in the agent's own rotated frame.

Scope of the defect:
- '6d' representation: 6 of 144 dims affected (root). Body joints are SMPL parent-relative =>
  frame-invariant, so 138 dims are fine.
- 'joints3d' representation: ALL 72 dims affected — _pose_features runs SMPL FK on
  [root_orient, pose_body] and removes only root TRANSLATION, so world-frame global rotation is
  baked into every joint coordinate. The F4 "SMPL vs skeleton keypoints" comparison was therefore
  confounded by the frame as well.

Why this was never caught: the AGENT_CENTRIC_POSE_ROT ablation (map_agentrot, 2026-07-21) and
ZERO_ROOT_ORIENT (2026-07-08/20) BOTH ran against the corrupt final_10fps, where the root
orientation belonged to a different pedestrian. Rotating a donor's orientation into the recipient's
frame is still meaningless, so those nulls carry no information about the frame question. The frame
hypothesis is UNTESTED on clean data.

Note the frame bug PREDICTS the observed correct-vs-shuffled null on the root channel: in world
frame both a correct and a donor orientation are un-anchored relative to an agent-centric
trajectory (the agent's world heading is not recoverable — it is subtracted out), so both are
equally unusable. It does NOT explain the whole null: the 138 frame-invariant body dims still
showed nothing.

Bounded expected payoff (measured, clean data):
- Moving pedestrians (speed>0.5 m/s, n=1500): corr(body-facing-minus-heading, future 3 s turn)
  = +0.068; left- vs right-facing quartiles differ by only ~4.5 deg of future turn. Weak.
- STANDING pedestrians who then walk >1 m (n=169 of 13,288 scanned, 1.3%): body facing predicts
  the direction of motion onset far above chance — within 30 deg for 43.8% (chance 17%), within
  45 deg for 56.8% (chance 25%), median error 35 deg.
=> the cue is REAL and is exactly what the current frame destroys, but it applies to ~1% of agents,
so a frame fix is cheap and worth running yet unlikely to overturn the population-level null.

### 2. Architecture integration — how pose enters today

Verified in mtr/models/context_encoder/mtr_encoder.py:
obj_poses (num_center, num_objects, 11, 144) -> GRU final hidden state OR cross-attention (agent
polyline feature as the single query, 11 pose tokens as keys/values) -> ONE D_MODEL vector per
agent -> concat with the agent polyline feature -> pose_fuser MLP (optionally zero-init residual)
-> replaces the agent feature -> global self-attention with map -> decoder.

Structural limitations this implies:
(a) temporal bottleneck: 11x144 collapses to one vector before any scene-level reasoning;
(b) late fusion: the agent polyline encoder that BUILDS the trajectory representation never sees
    pose — pose can only additively perturb a finished summary;
(c) decoder isolation: the 64 intention-point queries / 6 mode heads never attend to pose, so pose
    cannot directly disambiguate WHICH mode, only nudge the shared context;
(d) zero-init residual starts the branch at exactly zero contribution against an already-strong
    baseline;
(e) 144 raw dims at 6k-143k training pedestrians is a poor SNR trade — and we measured the branch
    itself costs +1.7% (p=0.0004) at the 25k rung, so any added-parameter design starts in a hole;
(f) no pose-pose interaction: each agent's pose is summarised independently; the social/interaction
    structure Waymo-3DSkelMo was built for is not modelled at all.
