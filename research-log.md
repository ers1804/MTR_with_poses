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

## 2026-08-26/27 — review follow-up launched (frame fix + architecture variant), first attempt lost to a wall-clock cap

Wave 1: AGENT_CENTRIC_POSE_ROT on clean data (2 LRs x 3 seeds) + rotated wrong-pose control (3 seeds).
Rotation verified before launch: center-agent root yaw circular std 2.045 rad (world, ~uniform)
-> 0.819 rad (agent frame), i.e. the body now faces roughly along travel.

Wave 2: 8 frame-INVARIANT descriptors (facing-vs-motion cos/sin, torso yaw rate cos/sin, hip-flexion
stride phase, cadence, arm swing, spine tilt) concatenated onto the agent's per-timestep features,
with the pose branch DISABLED (USE_POSE_ENCODER False, NUM_INPUT_ATTR_AGENT 29->37). This adds input
channels instead of an encoder, which matters because the branch itself measured +1.7% cost at 25k.
Validated: 4 unit tests incl. invariance to arbitrary global yaw (the first fixture was wrong -- a
pure yaw leaves body-forward vertical where yaw is undefined -- and the tests caught it); on real
data correct poses give median |body facing - heading| = 12.0 deg vs 93.4 deg for the shuffled
control, so for the FIRST TIME the control is genuinely distinguishable from the treatment (under
the world-frame encoding both were equally un-anchored, which is what the null was measuring).

FAILURE + FIX: all 18 tasks were cancelled at epoch ~21/30 by a 12 h #SBATCH cap. The runs were
healthy (0.16 s/iter, sensible losses); 25k-rung runs need ~16.5 h (12,500 iters/epoch x 30 epochs)
and the main 25k sweep only survived because it used 24 h. My earlier "~3 h/run" estimate for the
25k rung was wrong -- those runs took ~13 h; I had misread the submission time. Limit raised to the
24 h partition maximum and both waves resubmitted (790366 / 790367).

## 2026-08-27 — REVIEW FOLLOW-UP COMPLETE: frame fix does not rescue pose; early fusion removes the cost

18 runs on Helma (2 waves x 3 cells-ish x LRs x 3 seeds), 0 failures. Per-arm best LR = 2.5e-5 for
every cell, matching the 25k rung. Hierarchical bootstrap, n=3:

  reference 25k rung: nopose 0.3711 | branch-pose 0.3774 | branch-shuffle 0.3770

  WAVE 1 — coordinate-frame fix (AGENT_CENTRIC_POSE_ROT), pose branch retained
    fa_agentrot        0.3794 +- 0.0024
    fa_agentrot_shuf   0.3768 +- 0.0071
    correct vs its own control: -0.68%, hier p=0.37   -> NO content signal
    vs no-pose:                 +2.23%, hier p=0.0006 -> branch cost persists (and is not smaller)

  WAVE 2 — 8 frame-invariant descriptors, early fusion, NO pose branch
    fa_derived         0.3711 +- 0.0051
    fa_derived_shuf    0.3728 +- 0.0027
    correct vs its own control: +0.48%, hier p=0.65   -> NO content signal
    vs no-pose:                 -0.01%, hier p=0.9988 -> capacity tax GONE (exactly baseline)
    vs the branch design:       -1.66%, hier p=0.0056 -> early fusion is significantly cheaper

CONCLUSION. The coordinate-frame defect was real (verified: world-frame root vs agent-centric
trajectories; correct/shuffled facing separation 12 deg vs 93 deg after the fix, i.e. the control is
now genuinely different from the treatment), but correcting it does NOT produce a pose benefit, in
either the branch architecture or a branch-free early-fusion design. The informative cue exists in
the data and is now expressible; it still does not convert into trajectory accuracy.

The one positive, reproducible result: pose conditioning can be made FREE. Replacing the 144-d side
branch with 8 frame-invariant descriptors concatenated onto the agent features lands exactly on the
trajectory-only baseline (-0.01%, p=0.9988) and beats the branch design by 1.66% (p=0.0056). So the
+1.7% "cost of pose" measured at the 25k rung is an artifact of the ENCODER BRANCH, not of pose
conditioning as such.

This closes the architecture/representation line: content signal is absent under raw 6D, joints3d,
world-frame root, agent-centric root, high-dim branch encoding, and low-dim early fusion.

## 2026-08-29 — AVO harness implemented (Erik's request: agentic evolutionary search over pose integration)

Goal as stated: find a pose integration whose minADE beats no-pose. Framing agreed with the record:
the loop SEARCHES; any candidate it accepts is validated out-of-band (30-epoch recipe, n=5, shuffled
control, hierarchical bootstrap, eval-FINAL split) before being believed; stagnation is itself a result.

Design (Erik's choices: 25k corrected data, Helma-backed f, 2 fixed seeds, pose-overlay-only scope,
Sonnet nightly 10h):
- Search space = ONE file (workspace/src/overlay.py): derive_features() + optional fusion module.
  MODE "early" (features onto agent input channels) or "module" (custom fusion vs 256-d agent feat).
  Hook committed in the MTR repo (mtr/utils/pose_overlay.py + dataset/encoder branches); 7 unit tests
  incl. bit-identity of the seed overlay with the built-in 8 descriptors.
- f (harness/eval/score.py): local contract check -> rsync overlay -> 2 sbatch jobs against a FROZEN
  repo clone ($WS/MTR_avo @ 4016e12, verified per job, tracked-file cleanliness enforced) -> poll ->
  gates (sane range, dev-agent count, per-seed status) -> scores {seed101, seed202}, direction=min.
- Proxy recipe (search only, never paper numbers): 25k scenes, batch 8 on H100, 12 epochs, LR 1e-4,
  ~2.5 h/eval with seeds parallel -> ~4 attempts per 10 h night.
- Anti-selection-bias: loop scores ONLY the dev half of validation (2,617 scenes, md5-parity split,
  committed); the final half (2,554) is reserved for out-of-band confirmation.
- Agent constraints (AGENT.md): pose overlay only; no cluster contact; optimizer/data frozen; known
  dead ends and measured cues provided in knowledge/ so 4 attempts/night are not wasted rediscovering.

Calibration running (6 jobs): no-pose proxy baselines (s101/s202) + 2 repeats of the seed overlay per
seed -> pins BASELINE_DEV_MINADE constants and sets min_relative_gain above the measured repeat spread.

## 2026-08-30 — AVO night 1 was a PHANTOM (agent could not edit); two harness defects fixed

Journal forensics: all 3 "attempts" show the variation agent denied Edit/Write — claude 2.1.x
treats permission-mode `dontAsk` as deny-without-asking for edit tools (it IS a listed mode, so the
config parsed fine and the harness README's recommendation is stale for this CLI). The three
"rejections" were re-scores of the UNCHANGED seed. Fixed: bypassPermissions for both agent and
supervisor commands.

The phantom bought real metrology, though: 3 no-op evals through the loop path agree to ~0.1%
aggregate spread, but sit 0.5-0.8% BELOW the manually-submitted calibration runs on s101/s303.
Submission flags differ (loop: sbatch --export=ALL,AVO_LR=...; manual: header --export=NONE) and
correlate perfectly with the two eras. Whatever the mechanism, baselines measured through one path
are not comparable to scores measured through the other -> all three no-pose baselines re-measured
through the EXACT loop submission path (calbase2) before the search restarts. Floor stays 0.8%
(8x the within-path spread). Lineage journal archived (state/archive/journal_night1_phantom.jsonl);
night 2 restarts with a clean x0 under the fixed agent command.

Lesson for the paper's methodology section: even the measurement harness's SUBMISSION PATH is a
treatment. Two identical training jobs differing only in `sbatch --export` flags shifted minADE by
up to 0.8% — larger than every pose effect we ever measured.

## 2026-08-30 (cont.) — CORRECTION: no submission-path effect; s101 is simply a fat-tailed seed

The calbase2 re-measurement refutes my own "submission path is a treatment" claim from this morning:
s303 no-pose agrees to 0.07% across the two submission paths, s202 to 0.2%. What actually happened:
seed 101 has sigma ~1.5% (range 3.4% over 8 identical-code runs) REGARDLESS of code or path, and I
pattern-matched eras on n=3 samples of a fat-tailed distribution — the same class of error as the
project's seed artifacts, now committed by me at the metrology level. Retracted in place.

Operational state for the search (night 2, running):
- configs = seeds {202, 303} (sigma 0.3-0.5%); s101 dropped entirely
- loop-path no-pose baselines: s202 0.35044, s303 0.35541 (calbase2)
- floor 0.8% (~2.5x aggregate sigma), per-seed cap 1.5%
- agent permission mode fixed (bypassPermissions) — night 1 was a phantom (agent could not edit)
Incidental observation, not a claim: on the stable seeds the 8-descriptor seed overlay has run
~0.3-1.0% BELOW no-pose across 6 identical-code evals under the 12-epoch proxy. If the loop's x0
rescore confirms that, the search starts from a small real advantage under the proxy recipe.

## 2026-09-08 — AVO SEARCH CONCLUDED (stopping rule fired): no pose integration beat the incumbent

The agentic evolutionary search (NVIDIA AVO harness, Sonnet variation operator, Fable 5.1 supervisor)
terminated on its own pre-registered rule: 3 supervisor interventions produced no acceptance.

Setup: search space = the pose-integration overlay only (feature construction + optional fusion
module); frozen commit-pinned training repo; f = 25k-scene proxy (batch 8, 12 epochs), 2 stable
seeds in parallel on H100s, ~2.5-3 h per evaluation; acceptance = >=0.8% aggregate gain (>=2.5x the
measured noise) with no seed regressing >1.5%; loop scored ONLY the dev half of validation.

Result: 13 scored candidates, 0 accepted. Incumbent remained x0 (the 8 frame-invariant descriptors,
early fusion) at aggregate 0.3505.

  -0.49%  Tiled circular-mean body-facing + von Mises concentration   <- best; split seeds, below floor
  +0.02%  Gait-energy-gated body-facing (the standing-onset cue)
  +0.09%  Rotation-angle stride + tiled gait amplitude
  +0.42%  cos/sin maxpool decoupling fix
  +0.54%, +0.58%, +0.60%, +0.64%, +0.80%  (facing context, gait amplitude, head-gaze chain, ...)
  +0.98%  Descriptor pruning (8->6)
  +1.19%  Tiled aggregate facing context
  +1.44%  Module-mode last-frame FiLM gate            <- Fable-supervised final attempt
  +4.66%  Module-mode mean+max pooled fusion          <- worst; reproduces the branch capacity tax

Levers explored across the 13: temporal max-pool blindness (5 variants), body-facing concentration,
gait-energy gating, kinematic-chain features (head gaze, knees), angular encoding fixes, descriptor
pruning, and two module-mode architectures. The supervisor steered the last three.

Interpretation. A directed agent with the full dead-end list, the measured cue inventory, and 40+
H100-hours found nothing that clears a 0.8% floor on the dev split; the single best candidate
(-0.49%) fell to seed disagreement. Both module-mode designs reproduced the branch capacity tax
independently discovered in the manual study (+1.4% and +4.7%). This is a bound, not a proof: it says
that within this integration space, at this data scale, with this proxy, directed search does not
find a pose representation that helps. It is the natural closing experiment for the paper's null.

Cost accounting: ~$25 of agent reasoning, ~40 H100-hours of scored evaluations, plus 9 unscored
attempts lost to infrastructure (cluster reservations/NODE_FAIL/account limits) and 3 harness bugs
of mine (integrity trip, module-mode config keys, agent self-running bench.sh) - all annotated in
the journal so no hypothesis was recorded as refuted when it was never tested.

## 2026-09-08 — Event-conditioned re-analysis: the null holds inside every subpopulation

Motivation: the aggregate averages over 16.5k pedestrians, so a real effect confined to the ~1.5%
where the measured cue exists (standing pedestrians whose body facing predicts motion onset at 2.6x
chance) would be invisible. Re-analysis of the 25k triad's existing per-agent metrics (best epoch,
3 seeds, same hierarchical bootstrap as the headline numbers) — no training, no cluster.

Decisive comparison (correct pose vs its OWN shuffled control), by subpopulation:

  ALL                  n=16,342 (99.1%)   +0.08%  p=0.95
  standing_then_moves  n=267    ( 1.6%)   +2.39%  p=0.50   <- where the cue lives; correct is WORSE
  standing_stays       n=2,739  (16.6%)   +0.28%  p=0.72
  walking              n=11,956 (72.5%)   +0.19%  p=0.83

No subpopulation shows a pose-content benefit; the sign is positive (correct worse) in all four,
and the widest CI (standing_then_moves, +/-5% relative on n=267) cannot exclude a moderate effect
there but shows no hint of one. Note also the difficulty structure: standing-then-moves is the
hardest population by far (minADE ~1.29 vs 0.41 walking, 0.17 standing-stays) — exactly the
high-uncertainty regime where pose was hypothesised to help, and where it does not.

Secondary observation, consistent across every subpopulation: the pose BRANCH costs ~1.6% vs
no-pose (ALL p=0.0008, walking p=0.0056), while correct-vs-shuffled is flat — capacity, not content,
reproduced at subpopulation resolution.

Conclusion: the "the average hides it" explanation is now closed. Data: experiments/
subpopulation_pose_analysis.json; script: tools/scripts/subpopulation_pose_analysis.py.

## 2026-09-08 (cont.) — epoch-selection robustness check before re-using checkpoints

Only epoch-30 checkpoints survived the 25k runs, so an expanded-validation rerun would evaluate a
LAST-epoch operating point rather than the best-epoch one behind every headline number. Re-ran the
subpopulation analysis on existing metrics under both selection rules:

  decisive test (correct pose vs its own shuffled control):
      ALL                  best-epoch +0.08% p=0.95   |  last-epoch -2.49% p=0.58
      standing_then_moves  best-epoch +2.39% p=0.50   |  last-epoch +1.26% p=0.63
  => the pose-CONTENT conclusion is robust to epoch selection (null either way).

  cross-architecture comparison (no-pose vs pose) is NOT robust:
      ALL                  best-epoch -1.61% p=0.0008 |  last-epoch +8.00% p=0.30
  => at epoch 30 the no-pose arm has degraded far more than the pose arm. Late-epoch checkpoints
  must not be used to compare architecturally different arms; they are fine for comparing two
  structurally identical arms that differ only in pose content (pose vs shuffle).

Consequence: the expanded-validation rerun is scoped to the pose-vs-shuffle test only, and reported
as a supplementary high-agent-count check, not as a replacement for the best-epoch headline numbers.

## 2026-09-08 (cont.) — expanded-validation subpopulation test: 6.8x more agents, still null

Re-scored the nine epoch-30 checkpoints of the 25k triad on the FULL clean validation split
(27,156 scenes) — inference only, ~1.5 h of H100 time, no retraining. Evaluated agents went
16,342 -> 121,964; the rare standing-then-moves population went 267 -> 1,809 (6.8x).

Decisive test (correct pose vs its OWN shuffled control), epoch-30 operating point:

  ALL                  n=121,964   -2.65%  CI [-0.055,+0.013]  p=0.60
  standing_then_moves  n=  1,809   +0.38%  CI [-0.061,+0.058]  p=0.80
  standing_stays       n= 22,840   +0.66%  CI [-0.000,+0.003]  p=0.19
  walking              n= 86,214   -3.28%  CI [-0.078,+0.019]  p=0.60

standing_then_moves is the key row: with 6.8x the agents the point estimate collapses from +2.39%
(n=267, best epoch) to +0.38%, i.e. toward exactly zero, and the CI is now roughly +/-5% ABSOLUTE
on a mean of 1.19 (about +/-5% relative). No pose-content benefit where the measured cue lives.

Caveat kept explicit: these are LAST-epoch checkpoints, so the no-pose comparisons in this table
(+5% to +10%) are NOT trustworthy — the 2026-09-08 robustness check showed the cross-architecture
comparison flips sign under epoch selection while the content test does not. Only the content test
is read from this run. Data: experiments/subpopulation_pose_analysis_valfull.json.

With this, the "the aggregate hides a real effect in the population where the cue exists" hypothesis
is closed at ~7x the original resolution.

## 2026-09-09 — 25k rung at n=5: the decisive test is now the tightest measurement in the project

Seed extension complete (seeds 404/505 added at the tuned LR 2.5e-5), so the 25k triad is n=5:

  scale25k_nopose      0.3716 +- 0.0027
  scale25k_xattn_pe    0.3778 +- 0.0024
  scale25k_shufflepose 0.3777 +- 0.0053

  correct vs shuffled (DECISIVE):  -0.04%, hier CI [-0.0047,+0.0040], p=0.97
  no-pose vs correct:              +1.66%, hier p=0.0001
  no-pose vs shuffled:             +1.62%, hier p=0.0082

The content effect is now -0.04% with a hierarchical CI of about +/-1.2% relative — the tightest
bound the project has produced, and it straddles zero almost exactly. Compare the same test at
n=3: -0.09%, CI ~+/-1.8%. More seeds moved the estimate essentially not at all and shrank the
interval as expected for a true null.

The branch capacity tax is now unambiguous at n=5: BOTH pose arms cost ~1.6% versus no-pose
(p=0.0001 / p=0.0082) and are statistically indistinguishable from each other (p=0.97). Capacity,
not content — stated with the strongest evidence in the study.

Combined with the expanded-validation subpopulation test (121,964 agents; standing-then-moves
n=1,809, +0.38%, p=0.80), the two power upgrades Erik requested both landed on the same answer.

## 2026-09-10 — AVO round-2 candidate beats the incumbent on 5/5 seeds; decisive control running

The round-2 candidate "stationary-gated facing + position-based foot contacts" (eval 68bc3e7293,
recovered from the cluster after the harness discarded it) was re-scored against the incumbent x0
at three FRESH seeds, and both arms were run at all five so the comparison is paired:

  seed   candidate   x0        delta
   202   0.3502      0.3522    -0.57%
   303   0.3480      0.3513    -0.94%
   404   0.3531      0.3559    -0.79%
   505   0.3461      0.3505    -1.26%
   606   0.3540      0.3579    -1.09%

  paired delta -0.93%, sd 0.094pp, t(4) = -7.83, bootstrap CI [-1.13%, -0.72%], 5/5 seeds improved.

This clears the 0.8% acceptance floor and is the first candidate in either AVO round to improve
every seed. It requires the position block added on 2026-09-08 (foot contacts are not derivable
from rotations alone).

CRITICAL CAVEAT, and the reason no conclusion is drawn yet: beating x0 shows the FEATURES help,
not that POSE CONTENT helps. x0 and the candidate differ in feature engineering, so the gain could
be a better-conditioned input rather than information about the pedestrian's own body. The whole
project rests on the correct-vs-shuffled control, so the same candidate is now running at the same
five seeds with SHUFFLE_POSES_ACROSS_AGENTS=True (jobs 828266-828270, run from $WS/scripts so the
loop's frozen-repo pin is untouched).

  If shuffled also improves ~0.93% -> feature engineering, not content; the null stands.
  If shuffled does NOT improve      -> first genuine pose-content signal in the project.

## 2026-09-10 — DECISIVE CONTROL: the AVO round-2 candidate shows a POSE-CONTENT signal

The candidate (stationary-gated facing + position-derived foot contacts, eval 68bc3e7293) was run
at five seeds with SHUFFLE_POSES_ACROSS_AGENTS=True — identical code, identical pipeline, each
pedestrian receiving a DIFFERENT pedestrian's pose sequence. Proxy recipe, dev split, n_dev=8,445.

  seed   correct   shuffled    delta
   202   0.3502    0.3561     -1.66%
   303   0.3480    0.3551     -2.00%
   404   0.3531    0.3531      0.00%
   505   0.3461    0.3495     -0.97%
   606   0.3540    0.3565     -0.70%

  CONTENT TEST: -1.07%, sd 0.28pp, paired t(4) = -3.01, bootstrap CI [-1.68%, -0.47%], 4/5 seeds
  improved and none regressed.

  Reference: shuffled vs incumbent x0 = +0.14% (3/5) — the shuffled arm does NOT beat the
  incumbent, so the candidate's -0.93% gain over x0 is not explained by feature engineering,
  capacity, or regularisation. It tracks the pose CONTENT.

This is the first correct-vs-shuffled separation in the project. Every prior test, across six
representations, three data scales, two training regimes and 13 round-1 AVO candidates, returned
p >= 0.37 with the point estimate on zero.

STATUS: PROMISING, NOT ESTABLISHED. Caveats, stated before any claim is made:
  1. Proxy recipe only (12 epochs, batch 8, 25k scenes) — not the 30-epoch paper recipe.
  2. Dev split only; the eval-FINAL half has never been touched by this candidate.
  3. n=5 seeds; t=-3.01 is solid but the project has killed effects with better-looking numbers
     (the +13.4% top-1 artifact had p=0.0002 before the best-epoch correction).
  4. The candidate was SELECTED for being the best of ~9 round-2 candidates, so its estimate is
     upward-biased by selection; the confirmation must be treated as the first unbiased test.

NEXT (pre-registered here, before running): full confirmation protocol — 30-epoch recipe, n=5
seeds, correct vs shuffled vs no-pose, evaluated on the eval-FINAL split, hierarchical bootstrap.
Decision rule: the candidate is a real result only if correct-vs-shuffled is negative with
hierarchical p < 0.05 on eval-FINAL at the full recipe.

Recovery note: the control jobs trained correctly but my extraction step read the output directory
of the ORIGINAL config while the shuffle config wrote to its own; the numbers above were recovered
from the existing metrics without retraining.

### Epoch-selection robustness of the content signal (same day, before confirmation)

The 4/5 vs 5/5 asymmetry above prompted the check that killed the +13.4% artifact: the correct arm's
best epoch varies (10,12,12,12,11) while the shuffled arm always lands on 12, so best-epoch selection
could manufacture the gap. Re-ran the content test at every FIXED epoch, no selection:

  epoch    1     2     3     4     5     6     7     8     9    10    11    12
  rel%  +1.24 -2.76 -3.92 +0.07 -3.66 -1.33 -0.94 -2.33 -0.85 -1.54 -0.95 -0.70
  t     +0.50 -2.06 -1.87 +0.07 -3.08 -1.13 -1.23 -2.19 -4.25 -3.76 -1.48 -1.46

Negative at 11 of 12 epochs, and at every epoch of the converged region (9-12). So the signal is NOT
a selection artifact — but selection does inflate it. Honest bracketing of the effect size:

  best-epoch (as scored by the loop):  -1.07%, t(4) = -3.01
  fixed final epoch (12):              -0.70%, t(4) = -1.46   <- NOT significant at n=5

Both are reported; the fixed-epoch number is the conservative one and is what any claim must rest on.
Note also the trend within the converged region (-1.54% at ep10 -> -0.70% at ep12): the proxy recipe
truncates at 12 of 30 epochs, and the gap is shrinking as training proceeds. This is the single
biggest threat to the result and is precisely what the 30-epoch confirmation tests.

### HELD-OUT SPLIT: the content signal survives (same day, zero extra GPU)

The completed 12-epoch runs already contain per-agent metrics for BOTH halves of the validation
scenes, so the eval-FINAL split (2,554 scenes / 8,041 agents) — which the AVO loop has NEVER scored
and against which the candidate was therefore NOT selected — could be read straight off disk.

Protocol: epoch chosen on DEV, minADE reported on FINAL. No double-dipping.

  seed   correct        shuffled       rel
   202   0.3518 (ep10)  0.3560 (ep12)  -1.16%
   303   0.3483 (ep12)  0.3555 (ep12)  -2.03%
   404   0.3508 (ep12)  0.3561 (ep12)  -1.48%
   505   0.3497 (ep12)  0.3507 (ep12)  -0.28%
   606   0.3548 (ep11)  0.3587 (ep12)  -1.09%

  FINAL content test: -1.21%, t(4) = -4.24, 5/5 seeds improved, bootstrap CI [-1.69%, -0.68%]

Cross-checks:
  fixed epoch 12, no selection anywhere:  dev -0.70% (t=-1.46, 4/5) | FINAL -1.18% (t=-2.70, 5/5)
  negative at 11 of 12 epochs on FINAL.

The FINAL result is STRONGER and more consistent than the dev result (-1.18% vs -0.70% at the fixed
epoch; 5/5 vs 4/5). Selection bias produces the opposite ordering — dev flattering, held-out
deflating — so the gap cannot be explained by the candidate having been chosen on dev. This is the
first pose-content effect in this project that survives a held-out split.

REMAINING THREAT: all of it is the 12-epoch proxy, i.e. training truncated at 12 of 30 epochs, and
the dev-split gap narrows as training proceeds. If the effect is "correct poses converge faster"
rather than "correct poses converge better", 30 epochs will erase it. That is the confirmation now
launching: 3 arms (no-pose / correct / shuffled) x 5 seeds x 30 epochs, scored on both splits.
Decision rule, pre-registered: the effect is real only if correct-vs-shuffled stays negative with
p < 0.05 on FINAL at 30 epochs.

### Power analysis and seed extension to n=8 (2026-09-10, BEFORE any 30-epoch result existed)

Dry-running the confirmation analysis on synthetic data exposed two problems, both fixed before
the first CONF30 job produced a number.

1. The paired percentile bootstrap is anti-conservative at n=5. When all five per-seed deltas share
   a sign, a resampled mean essentially never crosses zero, so it returns p ~ 1/N_BOOT regardless of
   effect size. On synthetic data an effect with a true paired-t p of .053 was reported as p=.0001.
   This is the same failure mode that manufactured several already-retracted results in this project.
   FIX: the paired t-test p-value is now the gate; the bootstrap interval is kept only as a
   descriptive spread and its p is stored under the name p_boot_anticonservative.

2. n=5 is underpowered for the effect actually under test. Using the observed seed-to-seed
   variability of the paired correct-vs-shuffled delta on the FINAL split at fixed epoch 12
   (per-seed deltas -2.07, -2.03, -1.48, -0.28, -0.01 %; sd = 0.97 pp):

     power at alpha=.05      n=5     n=8     n=10
       effect 0.7%          23.7%   41.9%   52.7%
       effect 1.0%          41.9%   70.3%   82.3%
       effect 1.2%          55.1%   84.7%   93.3%
     minimum detectable effect at 80% power:  n=5 -> 1.64%   n=8 -> 1.13%   n=10 -> 0.97%

   The effect under test is ~1.2%, so n=5 would have coin-flip power and a null result would have
   been uninterpretable. FIX: seeds 707/808/909 added (jobs 829089-829097), taking the confirmation
   to n=8 per arm, 24 runs total.

   This is a power correction, NOT optional stopping: the decision was made from the 12-epoch
   variance while every 30-epoch job was still in its first epoch, so no confirmation outcome could
   have influenced it. The pre-registered primary endpoint and decision rule are unchanged
   (FINAL split, fixed last epoch, correct vs shuffled, paired t, p < 0.05).

Readout is pre-committed in tools/scripts/avo_confirmation_analysis.py, written before the data.

## 2026-09-11 — confirmation run 1 LOST to a tag-collision bug (mine), relaunched

The 24-job 30-epoch confirmation launched 2026-09-10 produced NO usable data. Two faults, the
first fatal:

1. FATAL (my bug in slurm/52_avo_confirm.sh): the job passed `--extra_tag "$TAG"` where TAG was the
   ARM name only ("CONF30_nopose"), with no seed. --extra_tag determines the training OUTPUT
   directory, so all 8 seeds of an arm wrote into ONE directory and overwrote each other's
   metrics_epoch_*.pkl. Only 3 output dirs existed where 24 were expected. The result JSON filenames
   DID carry the seed, which is why 4 of them looked "ok" — they were reading an interleaved
   mixture of 8 seeds and are meaningless. Verified by two different seeds (jobs 829040=s202,
   829043=s303) both logging `extra_tag CONF30_nopose`.
   FIX: TAG="${ARM}_s${SEED}".

2. 20 of 24 jobs hit the 20 h wall clock (one reached epoch 29/30 at 20:00:20). My ~7.5 h estimate
   counted training only. Training is ~15 min/epoch = 7.5 h, but eval-with-train fires on 21 of the
   30 epochs at ~38 min each, which is the real cost.
   FIX: --time=24:00:00, and --ckpt_save_interval 100 so the eval condition in
   train_utils.py:173 reduces to epochs {1,2,4} u {21..30} = 13 evals instead of 21. Estimated
   ~15.7 h. Epoch selection is now restricted to the converged region, which is if anything the
   more defensible protocol.

The 4 contaminated JSONs are kept at $WS/avo_confirm/archive_collided_20260910/ as evidence; the
collided output dirs (17 GB) were deleted. Relaunched as jobs 836067-836090, same 3 arms x 8 seeds,
same pre-registered primary endpoint and decision rule — nothing about the analysis changed, and no
confirmation result has been seen by anyone at any point.

## 2026-09-13 — CONFIRMATION: NOT CONFIRMED. The pose-content signal was a convergence artifact.

24/24 jobs COMPLETED (836067-836090), 3 arms x 8 seeds x 30 epochs, no collisions, no failures.
Verbatim output of tools/scripts/avo_confirmation_analysis.py (written before the data existed):

```
==============================================================================
30-EPOCH AVO CONFIRMATION -- pre-registered readout
==============================================================================
result dir : /tmp/avo_confirm
seeds      : [202, 303, 404, 505, 606, 707, 808, 909]
  nopose    last_epoch[30]  n_dev=[8445]  n_final=[8041]
  pose      last_epoch[30]  n_dev=[8445]  n_final=[8041]
  shuffled  last_epoch[30]  n_dev=[8445]  n_final=[8041]

per-seed minADE (FINAL split, fixed last epoch)
  seed         nopose        pose    shuffled
  202        0.3458      0.3506      0.3528
  303        0.3561      0.3510      0.3477
  404        0.3551      0.3514      0.3561
  505        0.3524      0.3536      0.3534
  606        0.3489      0.3510      0.3537
  707        0.3467      0.3565      0.3493
  808        0.3515      0.3555      0.3557
  909        0.3484      0.3600      0.3621

------------------------------------------------------------------------------
FINAL / fixed last epoch   <-- PRIMARY ENDPOINT
------------------------------------------------------------------------------
  pose     vs shuffled  -0.04%  CI [-0.67%, +0.69%]  t=-0.11  p=0.9159  5/8  n.s.
      (DECISIVE: pose content (capacity cancels))
  pose     vs nopose    +0.88%  CI [-0.20%, +1.96%]  t=+1.49  p=0.1790  2/8  n.s.
      (user-facing goal (confounded by architecture))
  shuffled vs nopose    +0.92%  CI [-0.26%, +2.06%]  t=+1.47  p=0.1862  1/8  n.s.
      (branch-capacity tax alone)

------------------------------------------------------------------------------
FINAL / epoch chosen on DEV
------------------------------------------------------------------------------
  pose     vs shuffled  -0.19%  CI [-0.86%, +0.51%]  t=-0.51  p=0.6289  5/8  n.s.
      (DECISIVE: pose content (capacity cancels))
  pose     vs nopose    -0.46%  CI [-1.32%, +0.37%]  t=-0.99  p=0.3556  5/8  n.s.
      (user-facing goal (confounded by architecture))
  shuffled vs nopose    -0.27%  CI [-0.73%, +0.24%]  t=-1.01  p=0.3479  5/8  n.s.
      (branch-capacity tax alone)

------------------------------------------------------------------------------
dev   / fixed last epoch
------------------------------------------------------------------------------
  pose     vs shuffled  +0.29%  CI [-0.18%, +0.68%]  t=+1.24  p=0.2537  2/8  n.s.
      (DECISIVE: pose content (capacity cancels))
  pose     vs nopose    +0.97%  CI [+0.14%, +1.88%]  t=+2.03  p=0.0819  3/8  n.s.
      (user-facing goal (confounded by architecture))
  shuffled vs nopose    +0.68%  CI [-0.40%, +1.91%]  t=+1.06  p=0.3251  3/8  n.s.
      (branch-capacity tax alone)

------------------------------------------------------------------------------
dev   / best epoch on DEV
------------------------------------------------------------------------------
  pose     vs shuffled  -0.29%  CI [-0.81%, +0.24%]  t=-1.02  p=0.3412  5/8  n.s.
      (DECISIVE: pose content (capacity cancels))
  pose     vs nopose    -0.44%  CI [-0.71%, -0.12%]  t=-2.71  p=0.0303  6/8  SIGNIFICANT
      (user-facing goal (confounded by architecture))
  shuffled vs nopose    -0.15%  CI [-0.74%, +0.49%]  t=-0.44  p=0.6732  4/8  n.s.
      (branch-capacity tax alone)

==============================================================================
PRE-REGISTERED VERDICT (FINAL split, fixed last epoch, correct vs shuffled)
==============================================================================
  effect -0.04%   p=0.9159 (paired t)   5/8 seeds improved
  --> NOT CONFIRMED. The 12-epoch proxy effect does not survive full training
      on held-out data; it was a convergence-rate artifact, not pose content.
      The null stands and the existing write-up remains correct.

  secondary (user-facing goal) pose vs no-pose: +0.88%, p=0.1790, 2/8
```

INTERPRETATION.

The pre-registered primary endpoint — FINAL split, fixed last epoch, correct vs shuffled — gives
-0.04% with p=0.92 and 5/8 seeds. The -1.18% observed at 12 epochs is GONE at 30. The hypothesis
flagged as the biggest threat when the run was launched turned out to be the right one: correct
poses converged slightly faster, not better. Once training completes, knowing whether a pedestrian
has its OWN body motion or a stranger's is worth nothing.

Every other honest cell agrees: the content test is -0.19% (p=.63) selecting the epoch on dev,
and +0.29% (p=.25) on dev at the fixed epoch. Pose-vs-nopose is +0.88% (p=.18) at the primary
endpoint, i.e. pose is if anything slightly WORSE, and the shuffled arm carries the same +0.92%
penalty — the familiar branch-capacity tax, paid regardless of pose content.

ONE CELL LOOKS LIKE A WIN AND MUST NOT BE REPORTED AS ONE: dev split / best-epoch-on-dev,
pose vs nopose, -0.44%, p=0.0303. That is the single most-selected cell in the table (epoch chosen
on dev, then scored on dev — double-dipping) AND it is the cross-architecture comparison the project
already knows is confounded by capacity. Its own control kills it: the content test in that same
cell is -0.29%, n.s. Quoting it would be exactly the cherry-pick the pre-registration exists to
prevent. It is recorded here only so nobody rediscovers it later and mistakes it for a result.

STATUS OF THE PROJECT'S HEADLINE CLAIM: the null stands, and is now better defended than before.
The paper and one-pager remain correct as written and need no revision. What changed is the
strength of the evidence: the null has now survived a directed agentic search (two rounds, ~22
scored candidates), a 216-d representation carrying explicit joint positions, its single most
promising candidate, n=8 seeds, the full 30-epoch recipe, and a held-out split — with the decisive
correct-vs-shuffled control at every stage.

The 12-epoch result was not fraud and not a bug; it was real at 12 epochs. It is a clean example of
why a truncated proxy recipe cannot settle a sub-1% effect, and belongs in the methodology section
alongside the other ten self-caught artifacts.

### Side finding from the confirmation data: early stopping helps, but it is an architecture effect

Asked of the 30-epoch confirmation: does stopping early help? Measured WITHIN the 30-epoch runs by
selecting the epoch on DEV and scoring on the held-out FINAL split (honest: dev-noise selection does
not transfer to a held-out split, so the measured gain is real rather than selection bias).

NOTE this is NOT the 12-vs-30-epoch comparison. Those are different LR schedules, each annealing to
zero at its own horizon, so they say nothing about early stopping.

  arm        early-stopped   epoch 30    gain
  no-pose      0.3487         0.3506    -0.55%  (p=.11, 6/8)
  pose         0.3471         0.3537    -1.87%  (p=.002, 8/8)
  shuffled     0.3477         0.3538    -1.71%  (p=.006, 7/8)

  median best epoch:  no-pose 26 | pose 22 | shuffled 23   (wider input peaks earlier)

Is the extra benefit a POSE-CONTENT effect? No.
  pose     vs no-pose  : extra benefit -1.32 pp (p=.055)
  shuffled vs no-pose  : extra benefit -1.17 pp (p=.072)
  pose     vs shuffled : extra benefit -0.15 pp (p=.790)   <-- null

The wider-input models overfit sooner and so gain more from stopping earlier, whether the poses they
receive are the pedestrian's own or a stranger's. Architecture, not information.

This explains an apparent inconsistency in the confirmation table: pose-vs-nopose is -0.46% when the
epoch is chosen on dev but +0.88% at the fixed final epoch. That sign flip is purely overfitting
timing; neither figure is significant (p=.36, p=.18) and the content control is null under both
protocols (-0.19%, -0.04%).

CAVEATS: evaluation covered only epochs {1,2,4} u {21..30} (a wall-clock compromise after run 1
timed out), so the true optimum may lie before epoch 21 and this gain may be UNDERSTATED. And
"best epoch on dev" is a realistic deployment protocol, not an oracle.

PRACTICAL: stopping around epoch 22-26 beats training to 30 by ~0.5% on the baseline and ~1.9% on
the wider-input variants — a larger effect than anything pose has produced in this project, and an
argument against fixed-epoch reporting generally. Worth a sentence in the methodology section.

## 2026-09-13 — ONSET TASK SCOPED: the null is EXPLAINED (pose facing is already in the dataset)

Scoped the motion-onset task as a possible positive-result direction. The task is well populated,
but the decisive test kills the pose hypothesis AND explains every null in this project.

DATA (validation split alone, deduplicated):
  19,939 unique stationary pedestrians (speed<0.3 m/s at t=0), 100% with SMPL pose.
  onset within 1/2/3/5 s: 0.3% / 3.8% / 8.6% / 16.8%  (67 / 753 / 1,706 / 3,340 events)
  NOTE: Waymo writes several sample_<scene>_<k>.pkl per scene carrying IDENTICAL tracks;
  raw extraction gave 107,847 rows for 19,939 real agents (5.41x). Deduplicate by
  (scene, object_id) or every count is inflated. Fixed in extract_onset_events.py.

(A) DETECTION — will a stationary pedestrian move >1 m within H s?  [test AUC, scene-split]
      H       kin   kin+head   kin+pose   kin+head+pose   kin+head+SHUFFLED pose
      2     0.741      0.735      0.745           0.739                    0.742
      3     0.703      0.694      0.697           0.704                    0.697
      5     0.638      0.636      0.651           0.646                    0.634
  Kinematics alone (residual micro-motion) carries the signal. Pose adds nothing, and the
  shuffled-pose control matches real pose everywhere. Null again.

(B) DIRECTION — which way do they step? (H=3 s, n=1,705 positive events)
      predictor            median |err|   <45 deg   <90 deg
      chance                      90.0      25.0%     50.0%
      SMPL body facing            44.2      50.6%     72.0%
      WAYMO box heading           25.2      63.9%     80.8%
  paired SMPL-minus-box = +7.3 deg median, Wilcoxon p=2.6e-43, SMPL better on only 35% of events.
  (the SMPL forward-axis convention was calibrated AGAINST the box heading first, which is
   deliberately generous to the pose side.)

INTERPRETATION — this is the mechanism behind the whole project's null:
  SMPL body facing IS informative (2x chance) — the earlier "2.6x chance" finding was real.
  But the Waymo tracker already estimates body orientation, ships it as the standard box-heading
  channel, and does so MORE ACCURATELY than LiDAR-HMR + SMPL fitting recovers it (25.2 vs 44.2 deg
  median error against actual onset direction). And MTR already consumes it:
  waymo_pose_dataset.py:1104-1105 feeds sin(heading)/cos(heading) into the agent input.

  So the one pose cue with predictive value was ALREADY AN INPUT to every model in this project,
  in a better form than pose could supply. Six representations, two AVO rounds and ~22 candidates
  were all competing against a channel the model already had. The null was overdetermined.

CONSEQUENCE: the onset task is NOT a viable positive-result direction as framed — both sub-tasks
are already solved by free features. Where pose could still win is precisely where the tracker's
heading is unavailable or unreliable (heavy occlusion, few LiDAR returns, long range, or datasets
without heading annotation). That is the next thing to test, by stratifying this same comparison
on range / point count / occlusion.

### Onset dataset, final scale (both splits extracted 2026-09-13)

  split        unique stationary peds   scenes    pose      onset<=2s   <=3s     <=5s
  training               224,946        101,338   99.4%       8,053    19,221   37,624
  validation              19,939          9,079  100.0%         753     1,706    3,340
  TOTAL                  244,885        110,417               8,806    20,927   40,964

Onset rates agree closely across the two splits (3.6/8.5/16.7% vs 3.8/8.6/16.8%), which is a
useful consistency check on the extraction. Raw row counts before deduplication were 1,232,196
and 107,847 (5.48x / 5.41x inflation from repeated sample_<scene>_<k>.pkl files).

So the dataset is NOT the limiting factor — ~245k stationary pedestrians and ~21k onset events at
a 3 s horizon. The limiting factor is that both sub-tasks are already solved by free features
(see above). Kept because it is the right testbed for the remaining open question: whether pose
beats the box heading in the regimes where the tracker degrades (occlusion, range, sparse returns).

### Stratified test: there is NO regime where SMPL facing beats the box heading (branch closed)

The last live hypothesis was that the box heading wins only where perception is healthy, and that
pose would take over in hard conditions. Tested by stratifying the same paired comparison
(H=3 s, n=1,705 positive onset events, validation) on three tracker-quality covariates.
Gap = pose-minus-box median angular error; NEGATIVE would mean pose wins.

  by RANGE TO EGO (m)          n     pose    box      gap        p     pose-wins
    Q1 [ 2.1, 16.9)          426     32.9   20.6    +12.3    7e-09       37%
    Q2 [16.9, 27.5)          426     40.6   24.3    +16.3    1.5e-09     37%
    Q3 [27.5, 40.6)          426     52.7   32.5    +20.2    2.8e-11     35%
    Q4 [40.6, 74.5)          427     60.2   27.3    +32.9    8.3e-19     31%

  by HISTORY VALID FRACTION (occlusion proxy)
    Q1 [0.09, 1.00)          309     60.2   26.5    +33.7    2.4e-11     32%
    Q4 [1.00]               1396     41.8   24.9    +16.9    2.1e-33     36%

  by BOX-HEADING JITTER (deg, circular std over history)
    Q1 [ 0.03,  2.99)        422     27.9   16.3    +11.6    1.2e-13     36%
    Q2 [ 2.99,  6.38)        422     38.8   19.2    +19.6    1.5e-15     33%
    Q3 [ 6.38, 13.62)        422     52.7   28.8    +23.9    1.6e-13     33%
    Q4 [13.62,124.70)        422     63.8   50.5    +13.3    3.2e-06     38%

The gap is POSITIVE in every stratum, all p < 1e-5. Worse for the hypothesis, it WIDENS with
range (+12 -> +33 deg) and with occlusion (+17 -> +34 deg): SMPL degrades FASTER than the box
heading exactly where pose was supposed to take over.

MECHANISM: both estimates are derived from the same LiDAR returns. When the returns are sparse,
distant or occluded, the mesh fit degrades at least as badly as the box orientation, so they fail
together rather than complementarily. Even in the jitteriest-heading quartile, where the box
heading is genuinely bad (50.5 deg median error), SMPL is worse still (63.8 deg).

CONCLUSION: the onset direction task offers no operating regime for pose. Combined with the
detection result (pose adds nothing over kinematics, shuffled control matches), the onset branch
is CLOSED. Body pose from LiDAR is dominated by a free channel everywhere it was hypothesised
to help, which is a complete and quantified explanation of this project's null.

### 2026-09-14 — closing gap: does pose ADD to the box heading for onset direction? No.

The 2026-09-13 direction test compared SMPL yaw ALONE vs box heading ALONE. That is the wrong
framing: the question is whether pose adds anything on top of the heading the model already has.
Closed here with a learned model, trained on the training split and tested on validation, using a
fuller pose descriptor (all three body-axis directions + articulation rates + last-frame joints)
rather than raw yaw.

  train positives 18,950 (training split) | test positives 1,688 (validation), H=3 s
  predictor                              median |err|   <45 deg
  raw box heading, no model                   25.1        63.8%
  model: box heading only                     24.8        64.1%
  model: box heading + POSE                   26.5        63.8%
  model: box heading + SHUFFLED pose          25.8        63.6%

  pose vs box-only     : +1.69 deg, Wilcoxon p=0.085
  SHUFFLED vs box-only : +1.01 deg, Wilcoxon p=0.214

Adding pose makes the prediction slightly WORSE, and adding SHUFFLED pose makes it worse by
almost the same amount — i.e. the degradation is the cost of extra feature dimensions, not a
property of pose content. Real-minus-shuffled is +0.68 deg, noise. A learned model over the full
pose descriptor does not recover anything that the raw-yaw comparison missed.

Both onset sub-tasks are now closed under the correct additive framing and with shuffled controls:
DETECTION (pose adds nothing over kinematics+heading) and DIRECTION (pose adds nothing over
heading). The stationary-pedestrian approach is fully tested and negative.

### 2026-09-14 — head orientation / gaze: NOT RECOVERABLE from this data (not merely unhelpful)

Tested whether head orientation (a gaze/attention proxy) can improve onset direction. Gaze
precedes locomotion in human motor control, so this is a genuinely different hypothesis from body
facing. It fails at the measurement level, before any modelling question arises.

DIAGNOSTIC — does the head move at all in the fitted meshes? (n=6,000 agents, last observed frame)
  joint            |axis-angle| mean    std
  neck                   0.0951       0.0588
  head                   0.1276       0.0546
  L_knee  (reference)      --         0.3216
  L_shoulder (reference)   --         0.1901
  head-vs-torso yaw: mean +0.8 deg, STD 3.8 deg, 5-95 pct [-4.4, +5.8]

The knee varies 6x more than the head. Real gaze spans roughly +-60 deg (people turn to look
before crossing); here the head is locked to the torso within about +-5 deg. LiDAR-HMR is emitting
its POSE PRIOR for the head, not a measurement.

PHYSICAL REASON: a pedestrian at 20-40 m in Waymo LiDAR is a few dozen returns. The head is a
handful of roughly spherical points with no geometry that constrains yaw — nothing in the point
cloud can distinguish face-forward from face-backward. The SMPL fit therefore falls back on the
prior, which centres the head on the torso. Gait is recoverable (the knees genuinely move); gaze
is not.

CONFIRMATORY TEST (n=1,705 positive onset events, H=3 s)
  residual to explain (onset dir vs body facing): std 66.9 deg
  head-vs-torso yaw dynamic range              : std  3.2 deg   (20x too small)
  correlation of head yaw with that residual: Pearson r=-0.033 (p=0.17), Spearman -0.021 (p=0.38)
     -> explains 0.11% of the variance, and the sign is wrong
  box heading alone                                  : median 25.2 deg
  box heading + head-yaw correction FIT IN-SAMPLE    : median 25.5 deg  (0.36 deg WORSE)
  Even fitting the correction on the test data itself — maximally generous, guaranteed to
  overfit in its favour — head yaw does not help.

CONCLUSION: head orientation cannot improve these results because it is not measured. This is a
sensor/method limit, not a behavioural claim: gaze may well matter, but LiDAR-based HMR at AV
ranges cannot recover it. Obtaining it would require camera-based gaze estimation, and WOMD ships
no raw images (tokens only), so that is only possible on the much smaller Perception split.

USEFUL COROLLARY for the MPJPE/dataset chapter: aggregate MPJPE hides which body parts are
actually recovered. Here the lower limbs carry real signal while the head is pure prior. Reporting
per-joint recoverability would be a more honest characterisation of LiDAR HMR than a single number.

## 2026-09-14 — Waymo PERCEPTION assessment: confirms the prior-collapse, cannot host the task

Asked whether the onset/pose experiments could be redone on Waymo Perception, which has camera,
LiDAR and GROUND-TRUTH 3D keypoints. Inventory of the local copy (/home/erik/raid/datasets/
waymo_perception), full dataset present:

  component            segments   labeled instances   distinct objects   frames/object (med/max)
  lidar_hkp  (3D GT)   798+202            9,998             2,917             2 / 41
  camera_hkp (2D GT)   798+202          173,526            16,188             6 / 85

  objects with >=10 labeled frames:  lidar_hkp 189   |   camera_hkp 5,721

(1) RESULT — GT CONFIRMS THE HEAD IS A PRIOR IN THE FITTED MESHES.
Using GT 3D keypoints (NOSE=1, HEAD_CENTER=20, L/R_SHOULDER=5/13; enum inferred from mean
keypoint height and validated by the forward-convention check: 95.6% of heads fall within 45 deg
of body-forward, 99.3% within 90 deg), head yaw relative to the shoulder line, n=8,234:

                            circular std   |yaw|>15   |yaw|>30   |yaw|>45   5-95 pct
  GROUND TRUTH                 19.2 deg      25.0%      10.0%       4.4%   [-30.8,+28.9]
  LiDAR-HMR fitted (WOMD)       3.8 deg      ~0%        ~0%         ~0%    [ -4.4, +5.8]

The real quantity varies 5x more than the fit reports. This confirms against ground truth that
LiDAR-HMR's head is regressed to its pose prior rather than measured. NOTE also that GT head yaw
is more modest than assumed: 75% of pedestrians hold their head within 15 deg of body-forward, and
only 1.9% exceed 60 deg. The earlier "gaze spans +-60 deg" expectation was too generous.

(2) CONSEQUENCE FOR THE ONSET TASK — Perception CANNOT host it.
The keypoint labels are frame-level annotations for pose-estimation benchmarking, not tracks:
median 2 labeled frames per object (LiDAR GT). Objects with the >=10 frames an 11-frame history
needs: 189 (LiDAR GT) and 5,721 (camera GT). At the measured 8.6% onset rate that is ~16 and ~490
onset events respectively — the latter the same order as the n=133 subgroup that was already
demonstrated hopeless (MDE 7.15% at n=8 seeds). So the trajectory/onset experiments cannot be
replicated there at usable power.

(3) HEADROOM EVEN WITH A PERFECT ORACLE.
The onset-direction residual that head yaw would have to explain has std 66.9 deg. GT head yaw has
std 19.2 deg. Even a PERFECT head-orientation oracle, perfectly correlated, covers under a third
of that spread (<10% of the variance). The ceiling on this cue is low regardless of sensor.

WHAT PERCEPTION IS ACTUALLY GOOD FOR: a per-joint recoverability study of LiDAR-based HMR against
GT (9,998 instances is well powered), i.e. "which body parts does LiDAR HMR actually measure?"
The head result above is one instance of it. That is a real, citable characterisation which
explains downstream failures for anyone building on LiDAR HMR, and it does not depend on pose
helping any downstream task.

### 2026-09-14 — HORIZON SWEEP: pose does not help at short horizons either

Hypothesis: gait phase determines the next step or two, so pose might help at 1-2 s even if intent
dominates at 8 s. Tested on the strongest data in the project (30-epoch confirmation, 3 arms x
8 seeds, held-out FINAL split, epoch 30, seed-paired), using the per-horizon metrics already
stored per agent. n = 8,038-8,041 agents.

  metric        nopose     pose     shuf    CONTENT (pose vs shuffled)      pose vs nopose
  minADE_2s     0.0911   0.0920   0.0916    +0.48%  t=+1.08  p=0.317 4/8     +1.09%  p=0.136
  minADE_4s     0.1728   0.1741   0.1737    +0.24%  t=+0.67  p=0.522 4/8     +0.74%  p=0.262
  minADE_8s     0.3506   0.3537   0.3538    -0.04%  t=-0.11  p=0.916 5/8     +0.88%  p=0.179
  minFDE        0.7551   0.7665   0.7668    -0.04%  t=-0.06  p=0.951 4/8     +1.52%  p=0.054

The content test is null at EVERY horizon, and its point estimate is if anything most POSITIVE
(i.e. pose slightly worse) at the SHORTEST horizon, decaying to zero by 8 s — the exact opposite
of the predicted pattern. pose-vs-nopose stays at the familiar branch-capacity tax throughout, and
that tax is largest on minFDE (+1.52%, p=0.054), the endpoint metric.

WHY THERE IS NO WINDOW: at 2 s minADE is 0.091 m. A pedestrian at 1.4 m/s covers ~2.8 m in that
time, so the model is already within ~3% of the displacement — short-horizon prediction is very
nearly saturated by linear kinematic extrapolation, leaving almost no headroom for any extra
signal. At long horizons the error is dominated by destination/intent, which body pose does not
reveal. Pose is squeezed out at both ends: the near field is already solved by velocity, the far
field is decided by goals.

That framing generalises beyond this project and is worth stating in the write-up: an auxiliary
modality only pays where the baseline is neither saturated nor dominated by unobservable intent,
and for pedestrian forecasting on Waymo that window is empty.

### 2026-09-16 — HIGH-DYNAMICS STRATIFICATION: null there too (and it sharpens the principle)

The horizon-sweep principle says a modality can only pay where the baseline is not saturated. The
sharpest operational version of "not saturated" is where CONSTANT-VELOCITY extrapolation fails, so
that is the best-motivated remaining test. Computed manoeuvre covariates for all pedestrians in
the eval scenes (13,982 peds; tools/scripts/highdyn_covariates.py) and stratified the content test
on the 30-epoch confirmation (8 seeds, held-out FINAL split, 6,764 agents with covariates).

CONTENT TEST (pose vs shuffled), by stratum, all four cuts:

  by CV RESIDUAL (m)      Q1      Q2      Q3      Q4      TOP 10%
    rel %              +1.19   -0.69   -0.28   +0.20     -0.13
    p                   0.104   0.469   0.680   0.777     0.797
  by HEADING VARIATION  -0.07   -0.07   -0.13   +0.01     +0.06   (all p >= 0.76)
  by SPEED RANGE        -0.80   +0.14   -0.10   -0.15     +0.56   (all p >= 0.45)
  by PEAK ACCEL         -0.02   -0.98   -0.00   +0.55     +0.79   (all p >= 0.14)

Twenty content tests; ZERO significant at p<0.05 (chance alone would give about one). There is no
stratum, on any manoeuvre axis, where correct pose separates from scrambled pose. In particular
the hardest CV-residual decile -- where the baseline has the most headroom by construction, mean
minADE 1.04 m -- gives -0.13%, p=0.80.

ONE CELL NEEDS FLAGGING SO IT IS NOT MISREAD LATER: in that hardest decile, pose-vs-nopose is
-1.28% (p=0.22), the only place pose looks better than no-pose. Its own content test is -0.13%
(n.s.), so this is the pose BRANCH's extra capacity helping slightly where the task is hard, not
pose information. Conversely in the easiest quartile pose-vs-nopose is +2.83% (p=0.032), the
branch tax showing up most where errors are smallest. Both are capacity effects with the same
null content test.

WHY THIS STRENGTHENS RATHER THAN WEAKENS THE PRINCIPLE: the principle has two clauses, and the
manoeuvre strata only relax the first (saturation). The second still binds -- when a pedestrian
deviates from constant velocity, the cause is typically EXTERNAL (another agent, an obstacle, a
kerb, a signal, a destination), not something written in their own body configuration. So
unsaturated headroom is necessary but not sufficient: the information governing the deviation
must also be observable in the modality. Pose fails the second clause even where it passes the
first. That is the sharper statement, and it is what the paper paragraph now says.

## 2026-09-16 — PER-JOINT RECOVERABILITY: MPJPE hides which joints are actually measured

New analysis (tools/scripts/per_joint_recoverability.py). A single MPJPE number cannot separate
"tracks the joint imprecisely" from "ignores the input and emits the prior". The second matters
downstream and is invisible in MPJPE, because a joint that barely moves scores well simply by
predicting the population mean. So each joint is scored against the strongest trivial competitor:
ALWAYS PREDICTING THE MEAN ROOT-ALIGNED GT POSE.

  SKILL = 1 - MPJPE / (error of the constant mean-pose predictor)
  SKILL > 0 beats the mean pose; SKILL <= 0 means the model adds nothing for that joint.

LiDAR-HMR stage, Waymo Perception validation, GT LiDAR keypoints, timestamp-matched
(index matching was artifact #10), zero betas, 347 matched frames over 113 pedestrians.
CIs are 2,000-sample bootstraps over frames.

  joint              n   MPJPE  meanpose    SKILL          95% CI   GTspr  PRspr  ratio  max|r|
  RIGHT_ELBOW      325    88.1     239.4   +0.632  [+.597,+.661]   239.4  242.1   1.01    0.93
  LEFT_ELBOW       323    93.7     235.4   +0.602  [+.559,+.639]   235.4  240.6   1.02    0.89
  RIGHT_HIP        347    39.3      94.8   +0.586  [+.557,+.609]    94.8   66.8   0.70    0.91
  LEFT_HIP         347    39.3      94.8   +0.585  [+.556,+.609]    94.8   66.7   0.70    0.91
  RIGHT_WRIST      328   135.7     276.8   +0.510  [+.457,+.558]   276.8  277.0   1.00    0.85
  RIGHT_ANKLE      344   105.3     203.8   +0.483  [+.434,+.528]   203.8  177.7   0.87    0.90
  LEFT_WRIST       316   128.9     248.6   +0.482  [+.429,+.531]   248.6  246.7   0.99    0.85
  LEFT_ANKLE       338   107.5     206.9   +0.480  [+.435,+.523]   206.9  177.6   0.86    0.90
  LEFT_SHOULDER    345    83.8     161.3   +0.480  [+.443,+.513]   161.3  175.8   1.09    0.91
  LEFT_KNEE        342    73.1     134.9   +0.458  [+.416,+.495]   134.9  122.0   0.90    0.85
  RIGHT_KNEE       344    74.9     137.9   +0.457  [+.416,+.491]   137.9  127.0   0.92    0.85
  RIGHT_SHOULDER   334    92.0     157.9   +0.417  [+.376,+.453]   157.9  180.3   1.14    0.91
  HEAD             322    97.6     100.1   +0.025  [-.036,+.078]   100.1   51.0   0.51    0.79
  OVERALL                 89.2     176.4   +0.494

(Overall 89.2 mm reproduces the published 87.6 mm; the small gap is the added <=60 ms match filter.)

HEADLINE. The HEAD is the only joint whose skill CI CONTAINS ZERO: it is statistically
indistinguishable from always predicting the average head position. Yet by MPJPE the head looks
MID-TABLE at 97.6 mm — BETTER than both wrists (136/129 mm) and both ankles (105/108 mm), which
carry skill ~+0.48-0.51. The head's flattering MPJPE is an artifact of its small GT spread
(100 mm, the smallest of any joint): when a joint barely moves, predicting the mean is already
nearly right. Its spread ratio of 0.51 (vs ~0.9-1.1 for limbs) shows the estimator additionally
under-dispersing it.

So MPJPE RANK-ORDERS THESE JOINTS ALMOST BACKWARDS with respect to information content. This
independently confirms, in the standard metric's own units and against ground truth, the
WOMD-side finding that the fitted head is the pose prior (head-vs-torso yaw: fitted std 3.8 deg
vs GT 19.2 deg).

LIMITATION: 347 matched frames from the 40 segments the pipeline was run on; 113 labeled segments
are downloaded, so this is roughly triplable with a GPU rerun of LiDAR-HMR + convert_amass. The
head-vs-limb gap is far larger than the CIs, so more data would tighten rather than change it.

### 2026-09-17 — 113-segment rerun: the head claim needed CORRECTING (and the finding survives)

Extended the recoverability evaluation from 40 to all 113 labeled+downloaded Perception segments:
347 -> 1,789 matched frames (5.16x), 113 -> 534 pedestrians. Overall 89.2 -> 91.1 mm,
skill +0.494 -> +0.516.

  joint              n   MPJPE  meanpose    SKILL          95% CI   GTspr  PRspr  ratio  max|r|
  LEFT_ELBOW      1646    91.2     252.9   +0.639  [+.623,+.655]   252.9  248.7   0.98    0.91
  RIGHT_ELBOW     1646    91.5     248.9   +0.632  [+.616,+.647]   248.9  247.4   0.99    0.91
  LEFT_HIP        1789    39.8      97.9   +0.594  [+.582,+.604]    97.9   68.5   0.70    0.91
  RIGHT_HIP       1789    39.8      97.9   +0.593  [+.582,+.604]    97.9   68.4   0.70    0.91
  LEFT_WRIST      1639   118.3     271.1   +0.564  [+.545,+.581]   271.1  259.7   0.96    0.88
  RIGHT_WRIST     1626   125.4     276.4   +0.546  [+.527,+.565]   276.4  274.6   0.99    0.87
  LEFT_SHOULDER   1753    87.3     171.5   +0.491  [+.474,+.507]   171.5  183.0   1.07    0.90
  RIGHT_ANKLE     1756   114.0     221.6   +0.485  [+.464,+.505]   221.6  190.7   0.86    0.85
  RIGHT_KNEE      1763    78.4     150.8   +0.480  [+.463,+.496]   150.8  137.8   0.91    0.85
  LEFT_ANKLE      1730   116.6     223.8   +0.479  [+.458,+.498]   223.8  188.1   0.84    0.85
  LEFT_KNEE       1748    81.5     151.2   +0.461  [+.444,+.478]   151.2  131.4   0.87    0.84
  RIGHT_SHOULDER  1750    89.0     164.0   +0.457  [+.441,+.473]   164.0  180.9   1.10    0.89
  HEAD            1688   111.6     116.9   +0.045  [+.022,+.067]   116.9   57.6   0.49    0.79
  OVERALL                 91.1     188.1   +0.516

CORRECTION TO THE 40-SEGMENT CLAIM. At n=347 the head's skill CI was [-0.036,+0.078], containing
zero, and I described the head as "statistically indistinguishable from always predicting the
average head position". At n=1,789 the CI is [+0.022,+0.067] and EXCLUDES zero. That claim was an
artifact of low power and is now retracted: the head carries a small but real amount of
information. The paper subsection and table have been updated; the "not measured" framing is
replaced by a quantitative one.

WHAT SURVIVES, AND IT IS THE ACTUAL POINT:
  - head skill +0.045 vs +0.457 to +0.639 for every limb joint: an ORDER OF MAGNITUDE less. The
    model cuts head error by 4.5% over the mean pose, against 46-64% for limbs.
  - spread ratio 0.49 vs 0.84-1.10 for limbs: the head is still heavily under-dispersed.
  - THE MPJPE INVERSION IS INTACT AND CLEANER: head MPJPE 111.6 mm is BETTER than both ankles
    (114.0 / 116.6) and both wrists (118.3 / 125.4), all of which carry ~10x its skill. MPJPE
    still rank-orders these joints backwards with respect to information content.

STABILITY OF THE REST. Five joints tripped a strict shift test (point estimate outside the other
run's CI), but every delta is between +0.002 and +0.082, all in the direction of HIGHER skill, and
no joint changed qualitative status. The flag fires because CIs tightened 2-3x (widths 0.05-0.11
-> 0.02-0.05), which makes the small-n estimates look slightly pessimistic rather than unstable.
The joint ordering and the head-vs-limb gap are unchanged.

LESSON, consistent with the rest of this project: an n=347 CI that straddles zero is not evidence
of absence. The 5x rerun cost ~12 h and converted a null claim into a quantitative one before it
reached a reviewer.

## 2026-10-19 — IV submission hardening: two wrong citations, a false CV claim, and a positive control

An Opus review of `paper/iv2027` for IEEE IV found no data problem (every number still traces to
the JSONs) but four blockers in scholarship and inference. All are now fixed; three of them
changed what the paper *claims*, not just how it reads.

**Citations.** `zhu2025waymo3dskelmo` sat in the .bib with zero `\cite` calls — the pose pipeline
the whole study runs on was uncredited, and the ICLR draft had cited it correctly, so this was a
regression introduced when cutting to 6 pages. Waymo Perception and its keypoint-label paper
(Zheng et al., CVPRW 2022) backed two of three contributions and were uncited. Two entries were
also simply wrong: Social-Transmotion carried arXiv **2312.11455** (an unrelated mathematics
paper; correct id is 2312.16168) and Social-Pose carried a stale title — both under a
"Source: verified via ..." comment, and the .bib header asserted every entry was fetched
programmatically. Header now says a comment is evidence of a check only when it carries a date.

**The constant-velocity claim was false.** §IV-D asserted short-horizon forecasting is "nearly
saturated by constant-velocity extrapolation", citing only MTR's own minADE as evidence. Measured
it (`tools/scripts/cv_baseline.py`, same validation loader, n=17,218 agents): single-mode CV ADE
is **0.310 m at 2 s** and **1.459 m at 8 s**, against the model's 0.091 m at 2 s. The model is
3.4x better than CV even at the short horizon — it is *not* CV-saturated. Reframed as a statement
about the forecaster's own residual (3% of displacement), which is the quantity that actually
bounds an auxiliary modality, and cited Schöller et al. (RA-L 2020) properly.

**The bootstrap leaked back in.** Table I quoted `p=0.0001`, which is exactly `1/n_boot` —
`_two_sided_p` floors there (`analyze_multiseed.py:259`). Every p-value and interval in the paper
is now a paired t over per-seed differences. Consequences: the branch tax on the shuffled arm is
**p=0.048**, not 0.008 (marginal, not comfortable); the headline bound widens from ±1.2% to
**±1.7%**; the n=8 MDE is **1.25%**, not 1.1%. The three-seed illustration got sharper rather than
weaker — those effects read significant under the pedestrian bootstrap (p=0.037/0.016) but give
p=0.18/0.12 under a seed-level t, so budget and inference unit mislead in the same direction.

**"Accepted none of 20 candidates" contradicted the next sentence.** Round 2 *did* accept a
candidate (that is why a pre-registration existed), and the count is ~22, not 20. Rewritten to
report the acceptance and let pre-registration be what removed it — a stronger story.

**Per-joint CIs were clustered wrong.** `per_joint_recoverability.py` bootstrapped *frames*
(n=1,789) rather than pedestrians (n=534). Added a pedestrian-clustered bootstrap. The head's
SKILL interval moves from [+0.022,+0.067] to **[-0.003,+0.089] — it now contains zero**, while
every limb interval still clears zero by a wide margin (lowest bound +0.43). This *restores* the
"head is not measured" reading, with a proper bound this time. Note the 2026-09-17 entry above
corrected the head framing in the other direction on the frame-level CI; the clustered interval
is the one to quote.

**Positive control (new arm, running).** The shuffled control shows the model cannot tell own-body
from stranger-body; it does NOT show the branch can transmit anything, since the fusion residual
is zero-initialised and a dead branch looks identical. Added `ORACLE_POSE` to the dataset
(`waymo_pose_dataset.py`): replaces SMPL features with a 6-channel encoding of the agent's own
future displacement at +3 s and +8 s, in the centre-object frame, same branch / parameter count /
input dim / validity masks. Verified the channel equals the unit future direction exactly and that
channels 6+ stay zero. Running n=3 seeds at the 579 rung (`tools/scripts/run_oracle_control.sh`).

Also added: minFDE throughout (content null there too: +0.33%, p=0.61 at 25k), a Protocol
subsection (hyperparameters, best-epoch selection, dev/held-out split sizes, why official mAP is
absent), disclosure that the ~1.6% branch tax is a best-epoch operating point which reverses at a
fixed late epoch, disclosure of the pipeline's root-orientation snapping, the image-HMR 106.2 mm
vs 87.6 mm comparison in Limitations, per-stratum n in Table III (and "twelve strata" -> ten: two
quartile cells are empty), and a Related Work paragraph engaging Social-Transmotion's JRDB
estimated-pose gain (0.40 -> 0.36) head-on.

### Positive control result (2026-10-19, 12:27)

The oracle arm (579 rung, n=3 seeds, best-epoch, same branch / parameter count / input dim /
validity masks as the pose arm, SMPL features replaced by a 6-channel encoding of the agent's own
future displacement at +3 s and +8 s):

  per-seed best minADE   0.2953 / 0.2916 / 0.2965   mean 0.2945
  vs no-pose  (0.4514 on the same 3 seeds)   -34.8%   t(2) = -81.4   p = 1.5e-4
  vs shuffled (0.4575)                       -35.6%   t(2) = -36.5   p = 7.5e-4
  vs pose     (0.4597)                       -35.9%   t(2) = -46.2   p = 4.7e-4

The reviewer objection "your branch is dead, that is why pose does nothing" is answered. The
pathway converts information into accuracy at ~20x the magnitude of the +/-1.7% bound the content
test places on pose. The null is about what SMPL pose contains, not about whether the architecture
can use what it is given.

Also measured while setting this up: pose-branch coverage is essentially complete -- 100% of
predicted agents have at least one valid pose frame and 86.7% of the 11 observed frames carry one
(n=7,828 centre agents). So "pose was rarely available" is not an alternative explanation either.
Both facts are now in the paper (Sec. II-A and Sec. III-E).

## 2026-10-19 (evening) — heading ablation: INCONCLUSIVE, not confirmatory. A power failure, not a result.

The mechanism counterfactual (zero the agent heading sin/cos channels, 579 rung, n=3 seeds,
best-epoch) completed 9/9. It does NOT support the paper's mechanism claim, and it does not refute
it either — the measurement cannot tell.

  per-seed best minADE
    nohead_nopose  0.4761 / 0.4452 / 0.4724   mean 0.4646
    nohead_pose    0.4767 / 0.4585 / 0.4683   mean 0.4678
    nohead_shuf    0.4551 / 0.4597 / 0.4959   mean 0.4702

  (i)   CONTENT without heading   -0.51%  t(2)=-0.12  p=0.91  CI [-13.2, +12.5]
  (ii)  pose vs nopose, no head   +0.70%  t(2)=+0.65  p=0.58
  (iii) cost of removing heading  +2.92%  t(2)=+2.45  p=0.13
  reference: CONTENT with heading, same 3 seeds  +0.48%  p=0.80

The sign of (i) is negative, which is the direction the mechanism predicts, and (iii) is positive
as expected — but the content interval is +/-13%, roughly eight times the effect being looked for.
Reading anything into the sign here would be exactly the error this project has already caught
three times.

WHY it is underpowered, which is the actually useful finding: removing the heading channel roughly
DOUBLES seed variance.

  arm sd over 3 seeds   nopose  with heading 0.0079 -> without 0.0169
                        shuffle with heading 0.0082 -> without 0.0223
  paired content sd     with heading 2.96pp -> without 5.16pp

Minimum detectable content effect at 80% power in the no-heading condition:
  n=3 15.97% | n=5 8.57% | n=10 5.13% | n=20 3.41% | n=40 2.34%

So brute-forcing seeds does not rescue this design — even n=40 (33 GPU-days at this rung) only
reaches 2.3%, and the target is 1-3%. The heading channel is evidently doing stabilisation work
beyond carrying facing information, which is itself worth knowing but is not what we set out to
measure.

NEXT, and it needs no new training: evaluate the existing checkpoints on the stationary-onset
subpopulation instead of overall minADE. The mechanism localises the effect to pedestrians who are
stationary and about to step off (8.5% of stationary agents move >1 m within 3 s); overall minADE
averages that cue away, which is a second reason the effect/noise ratio is hopeless above. A
targeted endpoint raises the effect without needing to lower the noise.
Caveat: `run_mechanism_queue.sh` deletes `best_model.pth` after each run to save disk, so only the
epoch-30 checkpoint survives. That is acceptable HERE because nohead_pose and nohead_shuf have
identical input dimensionality, so the fixed-late-epoch rule the paper warns about does not bite
for the content test (it would bite for pose-vs-nopose).

Status: the paper's Sec. IV mechanism argument still rests on the onset-direction and
stratification evidence, which is correlational. Do not claim the counterfactual supports it.

### Heading ablation, re-scored on subpopulations (2026-10-19, no GPU needed)

Correction to the entry above: `best_model.pth` is deleted by the queue script, but
`metrics_epoch_N.pkl` survives for EVERY epoch and holds one record per validation agent
(scenario_id, object_id, minADE, minFDE, horizon splits). So best-epoch re-scoring on any subset
needs no inference at all. `tools/scripts/nohead_subpopulation.py`.

16,342 agents common to all 6 runs (3 no-heading arms + 3 with-heading arms, n=3 seeds, best
epoch). Subpopulations: stationary (speed0 < 0.3 m/s) n=3,336; standing_then_moves (also
disp3s > 1 m) n=267.

CONTENT TEST (pose vs shuffled), by condition and subpopulation:

                        heading present      heading removed
  ALL                     +0.42%               -0.53%
  stationary              +2.73% (p=.067)      -1.43%
  standing_then_moves     +5.19%               -5.30%

The sign flips in every subpopulation once the heading channel is removed, and the magnitude
grows as the subpopulation narrows toward the one the mechanism names. Tested directly as a
difference-in-differences (content without heading minus content with heading, paired over seeds,
which is the actual hypothesis and avoids reading four separate CIs):

  ALL                  DiD  -0.82pp  t(2)=-0.32  p=0.78   per-seed [+1.16 +2.24 -5.86]
  stationary           DiD  -4.10pp  t(2)=-2.95  p=0.098  per-seed [-4.41 -1.56 -6.33]
  standing_then_moves  DiD  -9.85pp  t(2)=-1.75  p=0.22   per-seed [-14.84 +1.36 -16.05]

In the stationary subpopulation all three seeds agree in sign and the effect is where the
mechanism predicts. That is a HINT, not a result: p=0.098 at n=3 is exactly the regime where the
xattn+PE headline (-3.5% at n=3 -> -0.8% n.s. at n=5) and the AVO candidate (-1.18% at n=5 proxy
-> -0.04% at n=8 full) both dissolved in this project. Do not put it in the paper yet.

Caveats to carry forward: (a) three subpopulations were examined, so the stationary p=0.098 is not
a single pre-specified test; (b) the pre-specified target IS standing_then_moves, but it has only
267 agents on this validation split and its CI is +/-34pp; (c) the with-heading reference arms
already have n=5, so extending the no-heading triad to seeds 404/505 gives a paired n=5 DiD for
6 runs (~5.3 h). Power: DiD sd in the stationary cell is 2.41pp, so n=5 detects 4.0pp at 80%
power against an observed 4.1pp -- right at the edge -- and n=8 would detect 2.8pp.

## 2026-10-20 — detection floor: the pathway only pays for near-perfect direction (18/18 runs)

Direction-only oracle through the pose branch, corrupted by a fixed per-agent angular error,
579 rung, n=3 seeds, best epoch, paired t vs map_nopose_fix. Rungs labelled by the MEASURED
median absolute bearing error of the delivered channel.

  median |err|   rel minADE vs no-pose      p        95% CI
      0.0 deg      -15.74%               0.021    [-25.63, -5.89]   <-- only significant rung
     14.6 deg       +0.20%               0.876    [ -5.16, +5.60]
     27.2 deg       -0.10%               0.954    [ -5.35, +5.19]
     37.9 deg       +0.58%               0.624    [ -3.80, +4.97]
     45.0 deg       +2.72%               0.252    [ -4.61, +10.04]
     57.4 deg       +2.67%               0.237    [ -4.19, +9.50]

DETECTION FLOOR lies between 0 and 14.6 deg. Stated honestly: a perfect direction channel buys
-15.7%; by 14.6 deg median error nothing survives, and the interval there excludes gains larger
than 5.2%. The ladder cannot localise the floor more finely than (0, 14.6] at n=3 -- the per-rung
MDE is ~5-10pp, so a 2-3% gain at 14.6 deg would be invisible.

Two secondary observations, both coherent with the rest of the paper:
- The 0 deg rung gives -15.7% while the FULL oracle (direction + displacement magnitude) gave
  -34.8%. Dropping magnitude costs more than half the gain, as designed -- the ladder isolates
  bearing deliberately so the threshold can be stated in degrees.
- At 45.0 and 57.4 deg the branch is a net COST (+2.7%), matching the +1.6% branch-capacity tax
  measured at 25k. A sufficiently noisy side channel does not merely fail to help; it hurts.

WHAT THIS MEANS, AND THE TENSION IT CREATES
SMPL body facing is 44.2 deg (Table III). That sits in the region where the ladder shows a net
PENALTY, nowhere near the floor. So the null has a second, more self-contained explanation than
redundancy: pose fails the pathway's accuracy bar by a wide margin, whether or not a heading
channel exists. That also explains why the heading ablation was inconclusive -- removing heading
cannot help pose if pose was never accurate enough to pay.

But the box heading is 25.2 deg, which is ALSO below the floor. So the mechanism cannot be stated
as "pose loses because the tracker's heading is more accurate" -- through a side branch, neither
would pay. The defensible statement is narrower and needs care in Sec. IV:
  (a) a side branch must deliver bearing to better than ~15 deg to pay at all;
  (b) SMPL facing (44.2 deg) is far outside that, and is in fact in the penalty region;
  (c) the box heading is not subject to (a) because it is a NATIVE per-timestep input the model
      trains with from the start, not a late-fused side branch -- different delivery mechanism,
      not merely a different accuracy.
Do not write (c) as though it were measured: we have not run a ladder on the native channel.

NEXT (optional, 6 runs / ~5.3 h): rungs at ~5 and ~10 deg would localise the floor inside (0,14.6]
instead of leaving it a factor-of-three bracket. Worth it only if Sec. IV is going to quote a
threshold rather than a bracket.

## 2026-10-21 — heading ablation at n=8: the hint dissolved. Third one in this project.

24/24 extension runs, 0 failures. Paired seeds 101-808 in every cell (n=8).

Difference-in-differences (content without heading minus content with heading), n=3 -> n=8:

  subpopulation          n=3                    n=8
  ALL                    -0.82pp  p=0.78        +2.50pp  p=0.21   CI [-1.78,+6.78]
  stationary             -4.10pp  p=0.098       -0.58pp  p=0.65   CI [-3.45,+2.29]
  standing_then_moves    -9.85pp  p=0.22        -0.83pp  p=0.85   CI [-10.89,+9.23]

The stationary cell was the one with all three seeds agreeing in sign at n=3. At n=8 the point
estimate is one seventh of its former size and the sign of the ALL cell has flipped. This is a
PROPERLY POWERED null, not another underpowered one: the CI half-width in the stationary cell is
2.87pp, so the original -4.10pp is now excluded. The effect was not there.

THE COUNTERFACTUAL DOES NOT SUPPORT THE REDUNDANCY MECHANISM. Section IV's claim that pose is
redundant *because* the tracker heading is already an input remains correlational (onset-direction
error + stratification). Do not write it as causal.

This is the third signal in this project to dissolve with more seeds, after the xattn+PE headline
(-3.5% at n=3 -> -0.8% n.s. at n=5) and the AVO round-2 candidate (-1.18% at n=5 proxy -> -0.04%
at n=8 full recipe). The pattern is now so consistent it deserves to be the methodological point
of the paper: at this scale, n=3 produces sign-consistent effects of several percent that are
simply not real.

IT IS ALSO EXACTLY WHAT THE DETECTION FLOOR PREDICTS, which is the useful part. The decay law
gain(theta) = 16.0% * exp(-theta/6.41 deg) puts SMPL facing (44.2 deg) at a predicted gain of
0.016%. A channel that far below the floor cannot be rescued by deleting a competing channel --
there is nothing to un-block. The null DiD is a successful prediction of the floor model, not an
unexplained failure. The two results fit together:
  - detection floor: a late-fused side branch needs < ~10 deg bearing accuracy to pay at all
  - heading ablation: removing heading does not help pose, because pose was never above the floor

ONE CELL TO NOT OVER-READ: stationary / heading present / content = +1.79%, t(7)=3.41, p=0.0113,
i.e. correct pose significantly WORSE than shuffled pose. Twelve contrasts were computed (3
subpopulations x 4 tests), so this does not survive even a crude multiplicity correction
(Bonferroni 12 -> 0.13), and "correct pose is worse than a stranger's pose" has no mechanism
behind it. Treat as noise unless it replicates in a pre-specified test. Never quote it.

### Section IV rebuilt around the detection floor (2026-10-21)

Paper restructured so the floor carries the mechanism and redundancy becomes corroboration:

  IV-A "A detection floor for auxiliary channels" -- the ladder, the decay law
       Delta minADE(theta) = -16.0% * exp(-theta/6.41 deg), break-even at 13.3 deg,
       new Table II (8 rungs, CIs, paired t vs no-pose).
  IV-B "Pose is far below the floor" -- 44.2 deg measured on the onset task, law predicts
       0.02% against a ~2% capacity cost; box-heading comparison and the stratification
       demoted to corroboration that 44.2 deg is real.
  IV-B also carries "What this does not establish": the box heading at 25.2 deg is ALSO
       below the floor (0.31% predicted), so pose does not lose because heading is more
       accurate; the difference is delivery (native per-timestep input vs late-fused
       branch) and no ladder was run on the native channel. Then the n=8 heading-ablation
       null is reported as a successful prediction of the floor model rather than a
       failed experiment.

Also changed: the Introduction's second contribution is now the floor rather than the
redundancy mechanism (the old wording asserted as causal what the n=8 counterfactual does not
support); the Conclusion quotes the ~15 deg threshold; the general condition gains a third
clause (measurement error must leave the modality above the pathway's floor).

Cut to hold 6 pages: the onset-stratification table (folded into one sentence keeping the
range extremes and the jitteriest quartile), and compression of the Introduction, the tuning
paragraph, the practice section and Limitations. 6 pages, no overfull boxes, no undefined
references, 17 citations.

### 2026-10-23 — RANGE STRATIFICATION AT 25k: no near-field window either

The last untested regime. The floor says pose fails because facing is imprecise (44.2 deg pooled),
and facing improves with proximity (32.9 deg in the nearest onset quartile) because returns do
(median 329 points under 15 m vs 11 beyond 45 m). So if pose ever clears the floor, it is close in.

Scored the content test on range strata at the 25k rung from per-agent metrics already on disk
(tools/scripts/range_stratified_content.py; n=5 seeds, canonical best epochs from
multiseed_analysis.json -- two 25k seeds never got a best_eval_record.txt written). Range is
distance from the SDC at t=10, as in onset_stratify_covariates.py. 16,353 of 16,486 agents
(133 dropped with impossible ego distances).

Sanity check: ALL = +0.05%, CI [-1.63,+1.70], reproducing Table I (+0.06%, [-1.59,+1.70]).

  CONTENT TEST (pose vs shuffled), minADE, negative favours pose
    ALL            n=16353   +0.05%  p=0.94  [-1.63,+1.72]
    <19 m          n= 3594   -0.27%  p=0.65  [-1.80,+1.26]
    <15 m          n= 2324   +0.20%  p=0.76  [-1.46,+1.85]
    <10 m          n=  974   -0.14%  p=0.81  [-1.61,+1.33]
    Q1 [2,21)      n= 4088   -0.25%  p=0.67  [-1.73,+1.24]
    Q4 [51,75]     n= 4089   +0.44%  p=0.46  [-1.05,+1.94]

The near strata are nominally negative, which looks like a trace until the bins are made finer.
In 8 equal-count bins the sign alternates (+0.58, -1.20, -0.82, +1.13, -0.80, +1.07, +1.27,
-0.53) with nothing significant, and crucially the CLOSEST bin (1.9-13.9 m) is +0.58% -- pose
WORSE -- while the most negative is the second-farthest. Trend of content effect with range:
Pearson r=+0.20 (p=0.63), Spearman +0.43 (p=0.29). A genuine proximity benefit would show a
strong positive gradient; there is none. The "<19 m" number is a bin-edge artifact averaging the
+0.58% closest bin with the -1.20% next one.

Per-bin MDE is 2.0-4.1% at 80% power, so these bins exclude only effects above ~2%; the pooled
<19 m test excludes above ~1.8%. What they do rule out is a near-field effect large enough to
matter against the branch tax, which in the SAME stratum is +1.76% (p=0.005) -- significant, and
6x any content point estimate. Even a real -0.27% would be swallowed whole.

Consistent with the ladder a priori: at 32.9 deg (nearest onset quartile) the per-seed fits
predict a 0.02-0.71% gain, against that 1.65% tax. Theory and measurement agree.

DECISION: do NOT train on a near-only subset. The decision rule was "any trace of pose helping";
the finer-grained test says the trace is a binning artifact, the gradient is absent, and the
closest pedestrians favour a stranger's pose. Training on the subset would also cut the 25k rung
to ~22% of its agents, raising MDE well above the effect being chased.

NOTE: point count cannot be used as a stratifier here. The per-pedestrian return counts come from
Waymo Perception; WOMD motion data ships no per-agent LiDAR. Range is the available proxy.

### 2026-10-23 — POINT-COUNT STRATIFICATION AT 25k: still no window (the better cut, same answer)

Range was only a proxy. The pose pipeline's own extraction keeps the per-pedestrian point clouds
(scripts/extract_waymo_motion.py writes `pedestrian_pcd` per (object, frame); the files are on
the NAS at .../waymo_motion_extracted/validation/{scenario_id}.pkl). Return count governs
mesh-fit quality directly and absorbs OCCLUSION, which range cannot: a close half-occluded
pedestrian and a clear distant one can share a range and differ 10x in returns.

Built counts for all 16,486 25k-validation agents (tools/scripts/make_pointcounts_25k.py;
16,486/16,486 resolved, 0 scenario files missing). Mean returns over the 11 observed frames:
p10=14, p25=23, p50=48, p75=115, p90=275. Range vs returns: Spearman -0.87, so related but not
the same cut.

CONTENT TEST (pose vs shuffled), minADE, n=5 seeds, negative favours pose:
  ALL          n=16486   +0.06%  p=0.93  [-1.59,+1.70]   (reproduces Table I exactly)
  >=30 pts     n=10902   -0.08%  p=0.91  MDE 2.50%
  >=50 pts     n= 8083   -0.21%  p=0.78  MDE 2.63%
  >=100 pts    n= 4722   +0.06%  p=0.94  MDE 2.86%
  >=200 pts    n= 2372   -0.37%  p=0.54  MDE 2.02%
  Q4 (>=115)   n= 4124   -0.20%  p=0.80  MDE 2.84%

Across 8 equal-count bins the effect wanders (+0.59, -0.06, +0.81, +0.62, -1.06, +0.46, -0.02,
-0.37) with nothing significant. The bin-level trend is nominally in the PREDICTED direction for
once -- Pearson r=-0.38 (p=0.35), Spearman -0.45 (p=0.26) -- so it was worth testing properly
rather than eyeballing 8 points.

DECISIVE TEST: per-agent OLS of (pose - shuffled) minADE on standardised log10(returns), slope
taken per seed, then a seed-level t on the five slopes (the project's standard gate):

  seed 101 +0.00313 | 202 -0.00096 | 303 -0.00173 | 404 -0.00160 | 505 -0.00400  (m per SD)
  mean = -0.00103 m/SD   t=-0.89   p=0.423   CI [-0.00426,+0.00219]

Null, and one seed has the opposite sign. Even taking the point estimate at face value: -0.001 m
per SD on a 0.38 m baseline is 0.27% per SD of log-returns, i.e. moving a pedestrian from the
median (48 returns) to the top decile (275) buys ~0.27% -- against a branch tax of +1.66%
(p=0.0005) in the same runs. Six times under water on the most favourable reading available.

DECISION: no near-only and no well-resolved-only training run. The decision rule was "any trace
of pose helping"; on the cut that most directly measures how well the body was MEASURED, the
trace is absent at the gate and economically irrelevant even at its point estimate. Subsetting to
>=200 returns would also cut the rung to 14% of its agents.

This closes the last regime the detection floor left open. Pose does not pay anywhere we can cut
the data: not by manoeuvre (20 tests), not by horizon, not by range, not by how well the sensor
resolved the body.

### 2026-10-23 — LONGER OBSERVATION WINDOW: blocked by the dataset, and would not be enough

Q: would more observed pose frames make body facing accurate enough to clear the floor?

HARD BLOCKER FIRST. WOMD's LiDAR covers exactly the 1 s history: the extracted scenario pickles
hold 11 frames spanning 1.00 s (timestamps 0 .. 999940 us) out of a 9.1 s scenario. There is
nothing to extend into -- poses cannot exist outside that window, so the horizon cannot be
increased on WOMD at all. The 11 frames are the dataset, not a design choice. (The existing
window is not even full: across the export the mean is 8.9 of 11 frames, with only 56.7% of
agents at the full 11.)

That incompleteness is a natural experiment for the same question, so it was used as one
(tools/scripts/facing_vs_observation_length.py, reusing the calibration of
onset_pose_vs_heading.py, which is generous to pose). Reproduces the established figures exactly
on all 1,705 onset events: SMPL facing 44.2 deg, box heading 25.2 deg.

  frames   n      SMPL med   box med
  1-3      41      77.0        16.5
  4-6      61      70.6        22.2
  7-9     130      46.7        24.5
  10       77      62.4        27.8
  11     1396      41.8        24.9
  Spearman(frames, facing error) = -0.079, p=0.001

So more frames DOES help -- 77 deg at 1-3 frames to 41.8 deg at 11, nearly halving. But the
scaling is weak. A sqrt(n)-weighted power-law fit gives

  facing error = 117.4 * frames^(-0.423)      (predicts 42.6 deg at 11 frames; measured 41.8)

Extrapolating to the floor bracket:
  to 10.2 deg (top of bracket):   324 frames =  32 s of continuous per-pedestrian observation
  to  5.5 deg (bottom):         1,398 frames = 140 s
  a 3x window (3 s)  -> 26.8 deg   |   a 10x window (11 s) -> 16.1 deg

Both intermediate cases are still outside the bracket. You would need roughly THIRTY SECONDS of
unbroken LiDAR on a single pedestrian to make this cue pay through this pathway; WOMD gives one.

CAVEATS: the extrapolation runs ~30x beyond the data and rests on 5 group medians, one of which
(10 frames, 62.4 deg) is out of line. More importantly the comparison is NOT causal -- agents
with few frames are also the occluded and distant ones, so part of the gain with frame count is
selection. That makes the fit an UPPER BOUND on what a longer window would buy, which only
strengthens the conclusion.

NOTE: the 30fps arm already in the matrix is NOT this experiment -- it varied future-pose
SUPERVISION (giving the auxiliary losses real gradients), not the observation window.

A STALE-DATA TRAP worth recording: /home/erik/big1/waymo_motion_poses/final_10fps is the
CORRECTED per-scene re-export; the NAS copy at .../generated_motion/final_10fps is the
pre-fix one. Running this analysis against the NAS path loaded only 103/1705 events and gave
84.8 deg instead of 44.2. Always check a facing analysis reproduces 44.2/25.2 before trusting it.

### 2026-10-23 — FACING ERROR IN THE LADDER'S UNITS: the 44.2 deg comparison was not apples-to-apples

The review's M2, acted on. The ladder's theta is the error on FUTURE DISPLACEMENT BEARING at
+3/+8 s over every predicted pedestrian. The paper's 44.2 deg is onset direction for the
STATIONARY subset. Comparing one to the other is a category error, so this measures SMPL facing
against the ladder's target on the ladder's population
(tools/scripts/facing_error_ladder_units.py; 579-rung validation, 7,620 agents with a pose
sequence and valid +3/+8 s states; body-forward calibrated against the box heading, generous to
pose; calibration lands at 11.5 deg median vs box).

  target                       n      SMPL     box    SMPL >10.2 deg   box >10.2 deg
  +3 s displacement bearing   5824    12.3     3.9        56.8%            22.3%
  +8 s displacement bearing   6206    15.1     7.1        63.6%            40.8%
  percentiles (+3 s): SMPL p50 12.3 / p75 23.9 / p90 45.9 ; box p50 3.9 / p75 9.3 / p90 20.1
  moving (>=0.3 m/s)          5562    11.9     3.7
  stationary (<0.3 m/s)        262    41.2    24.7     (reproduces the 44.2/25.2 story)

TAKEN NAIVELY THIS BREAKS THE PAPER'S ARGUMENT: at a median of 12.3 deg the per-seed ladder fits
predict a 1.46-4.11% gain, which MEETS OR EXCEEDS the ~1.6% branch cost. Written as "44.2 deg,
far below the floor", Sec. IV-B is comparing the wrong number.

IT DOES NOT SURVIVE SCRUTINY, FOR TWO REASONS, AND THE RESTATED VERSION IS STRONGER.

1. The median is the wrong statistic. A ladder rung corrupts EVERY agent by about theta, so a
   channel is comparable to a rung only if its whole distribution sits there. SMPL's does not:
   56.8% of agents are outside the 10.2 deg bracket at +3 s and 63.6% at +8 s, with p90 at 45.9
   and 59.6 deg. A channel that is excellent for half the agents and useless for the other half
   is not a 12 deg channel.

2. It is mostly a copy of an input the model already has. SMPL facing sits 11.5 deg from the box
   heading in median, which is the pipeline's root-orientation snapping (Sec. III-A) showing up
   exactly where predicted: for movers -- 95% of this population -- body facing is derived from
   the trajectory. And the box heading beats SMPL at EVERY percentile (3.9 vs 12.3 median; 22.3%
   vs 56.8% outside the bracket) while already being a per-timestep model input. The oracle at
   theta is 12 deg accurate about the FUTURE; SMPL at 12 deg is accurate about the PRESENT
   heading, which the model is already given, and better.

This also squares with the n=8 heading ablation (-0.6 pp, p=0.65): pose is not merely redundant,
it is ALSO below the floor for most agents, so removing the competing channel cannot rescue it.
Redundancy and insufficiency both hold; neither alone is the whole mechanism.

PAPER CONSEQUENCE: Sec. IV-B must stop asserting "44.2 deg, far below the floor" as the headline
comparison. The defensible claim is the share-of-agents one: on the ladder's own target, the
majority of pedestrians fall outside the bracket, and the cue is a noisier copy of a channel the
model already receives. The 44.2 deg figure stays, correctly labelled as the stationary-onset
number, where it is the right measure of the one case pose was supposed to serve.

### 2026-10-23 Sec. IV-B rewritten to the ladder's own units

Applied the consequence above. Sec. IV-B is now "Where body facing sits relative to the floor",
and leads with the honest measurement (12.3 deg at +3 s, 15.1 at +8 s, on the ladder's target and
population) including the fact that it naively predicts 1.5-4.1% -- at or above the branch tax --
before decomposing it three ways: 95% of the scored agents are movers whose root orientation is
snapped to the direction of travel (stationary subset 41.2 deg, predicting 0.003-0.35%); a rung is
a uniform corruption while SMPL is a heavy tail (56.8% / 63.6% outside the bracket); and the box
heading is better at every percentile (3.8 deg pooled) while already being a per-timestep input.

The old "factor of four in bearing error outside the bracket" sentence is gone -- it was an
artifact of reading an onset-direction number against a displacement-bearing floor. 44.2 deg
survives in exactly one place, labelled as the stationary-onset measurement, where it corroborates
the 41.2 deg figure.

Two downstream claims also needed the units named. The "What this does not establish" paragraph
asserted the box heading is "also outside the bracket"; that is true on the stationary subset
(25.2 deg) but false pooled (3.8 deg, well inside), so the paragraph now says both and keeps the
point it was making -- that through a late-fused branch neither accuracy would pay, and that the
floor is a property of the branch, not of every delivery mechanism. The point-cloud figure caption
no longer cites 44.2 deg as the pooled facing accuracy.

Builds clean, 8 pages (limit is 6; cuts deferred by request).

### 2026-10-23 detection_floor.py hardened before the n=5 runs land

Two things that would have corrupted the extension, found by running the analysis against a
queue that had been going for 40 minutes.

1. THE SCRIPT SCORED A RUNNING JOB. best_minade() fell back to min() over every minADE in the
   /tmp log, and best_eval_record.txt is appended after EVERY eval epoch, so neither its presence
   nor the log told the script the run had ended. Rung 10.2 read n=4 and +3.97% because seed 404
   was 2/30 epochs in and contributed +25.0%. It now requires a DONE marker or a record that has
   reached epoch 30, and prints what it skipped. The n=3 numbers reproduce Table II exactly
   (MDE 12.30/0.84/6.17/6.71/6.57/5.46/9.13/8.53), so the published values are unaffected.

2. map_nopose_fix_s404 is a ~2 SD outlier FOR ITS CELL: 0.4954 against 0.4428-0.4584, where the
   sibling cells map_xattn_pe_fix and map_shufflepose_fix put s404 at 0.4570 and 0.4547, and seed
   404 averages BETTER than its cell mean across 37 cells (within-cell z = -0.37). So it is a bad
   run, not a bad seed, and it is the paired denominator for two of the four new comparisons.

   KEPT, not re-rolled. A baseline landing badly is exactly the training noise the seed-level test
   is meant to absorb, and re-running it to get a more convenient denominator is selection. It
   also errs the safe way: a too-high baseline inflates the oracle's apparent gain at rungs 10 and
   15, which makes it HARDER to claim the gain is gone by 10.2 deg. The script now prints a
   leave-one-out range per rung, and names seed 404's contribution explicitly, so no rung's
   verdict can rest on it unnoticed.

Reference points in the script were also moved into the ladder's own units (pooled box 3.8 deg,
SMPL 12.3; stationary box 24.7, SMPL 41.2) with a note that a median is not a rung, so the
mixed-units comparison cannot creep back in.

### 2026-10-23 review of the queued runs; bracket inference rebuilt

A Fable review of the launched Part A / Part B setup found nothing that invalidates either, and
verified the thing I most wanted checked independently: the early-fusion path delivers a tensor
BIT-IDENTICAL to the late-fusion one at a given rung (early obj_trajs[..., 29:35] == cross-attn
obj_poses[..., :6], checked at rungs 00 and 10), the first 29 channels equal the no-pose input,
every other batch key is equal, the noise is applied BEFORE the carry-through (rotate pairs (0,1)
and (3,4), then dir-only zeroes channels 2 and 5), masks carry over, and rung 00's direction equals
each agent's own future displacement bearing for 74/74 agents. So the rungs keep their measured-
error labels across pathways, and Part B is a valid comparison against map_nopose_fix: the configs
differ from the baseline only in the fusion keys, the channel count 29->35, and pose loss weights
that are inert (2 of 30,960 target steps have a valid future pose). Also confirmed the pose branch
is genuinely off (USE_POSE_ENCODER False skips construction, so no second route and no unused
parameters), DONE cannot be written for a crashed run, and the queue's output path matches the one
train.py builds.

THREE REAL FINDINGS, all in the analysis rather than the experiment, all fixed:

1. THE BRACKET RULE DID NOT SURVIVE MIXED n. It was max(significant) / min(non-significant). With
   the bracket rungs going to n=5 while the rest stay at n=3, and MDE scaling as 1/sqrt(n), a
   higher rung can read significant purely because it is better powered -- and max/min then yields
   an INVERTED bracket, e.g. [14.6, 10.2], which the reference-point test reads as "pays" because
   it checks th < lo first. At pooled SMPL 12.3 deg that would have flipped the paper's headline
   from "outside the floor" to "inside" on the strength of one rung's power. Replaced with the
   first monotone significant->n.s. transition, with any later significant rung reported as an
   anomaly instead of folded into an edge. Lifted to module scope and exercised on five cases
   (monotone, non-monotone, all-sig, none-sig, unsorted); the non-monotone case now returns
   [5.5, 10.2] + anomaly at 14.6 where the old rule gave [14.6, 10.2].

2. THE BRACKET IS NOW REPORTED BOTH WITH AND WITHOUT s404, and shouts if the edge moves. The
   review sharpened my earlier note: s404's high baseline is conservative for the claim "the gain
   is gone by 10.2 deg", but it biases the BRACKET LOCATION upward, and the bracket is itself a
   headline number. If the s404 oracle run lands at a cell-typical 0.45 its pair is about -9.2%,
   enough to flip rung 10.2 significant on its own. Currently both readings give [5.5, 10.2].
   (Mid-run, s404's oracle is tracking 2-4% better than its own baseline at matched epochs, i.e.
   seed 404 looks slow in both arms, which would make the pair well-behaved -- but that is a
   snapshot, not a result.)

3. PART B HAD NO CONSUMER: the script hardcoded the cross-attention config and MS_oracledir tags,
   so scoring the early ladder would have meant an ad-hoc path that bypassed the completion gate --
   the exact bug fixed earlier today. Now parameterised: `detection_floor.py [late|early]`.

Minor, also fixed: the printed `rel` column was ratio-of-means while t, p, CI, MDE and the JSON all
used mean-of-ratios. Table II in the paper uses mean-of-ratios (its 0.0 deg row reads -15.8%, which
only that definition gives), so the print was the odd one out; all eight rows now agree exactly.

DEFERRED, with a reason. The review notes Part B's motivation -- "early fusion carries no capacity
tax" -- was measured at 25k scenes and LR 5e-5, while Part B runs at 579 scenes and LR 1e-4, so
rung 00 alone cannot separate "pathway live" from "tax at this scale". A 3-seed early arm with the
oracle channel shuffled across agents would isolate it. Not queued yet: the floor is read off the
zero-crossing across rungs, which already absorbs whatever the tax is, so the measurement stands
on its own. The control is only needed to EXPLAIN a disagreement between the two pathways' floors,
so it is worth running if and only if they disagree.

### 2026-10-23 rung 10.2 at n=4: the s404 denominator dominates it

MS_oracledir10_s404 finished (52 min/run, so the 13-run queue lands around 23:00). Its pair reads
-15.45% against -2.38 / -2.75 / -6.00 for the other three seeds. The rung mean moves from -3.72%
to -6.65%, but the variance inflation more than cancels it: MDE goes 6.2 -> 12.7 pp and p from
0.084 to 0.117. The bracket is unchanged at [5.5, 10.2] under both denominators -- by luck, not
by design. (The review's mid-run read that s404 was "tracking 2-4% better than its baseline at
matched epochs" did not hold at the final epoch.)

POWER, WHICH DECIDES WHETHER PART A CAN WORK AT ALL:
  with s404     mean -6.64%  sd 6.09  -> needs n=9; n=5 gives MDE 10.1%
  without s404  mean -3.71%  sd 1.99  -> needs n=5; n=5 gives MDE 3.3%
So Part A's whole premise -- that n=5 converts the bracket into a measurement -- holds if and only
if that one denominator is sound. And it is load-bearing for the paper: if rung 10.2 goes
significant the bracket's lower edge moves 5.5 -> 10.2, which puts pooled SMPL facing (12.3 deg)
INSIDE the bracket and makes Sec. IV-B's three-way decomposition (heavy tail, trajectory-derived
median, noisier copy of the box heading) the load-bearing argument rather than a supporting one.

DECOMPOSING THE -15.45%: the s404 oracle run landed at 0.4189, which is BETTER than the other
three oracles (0.4346 average), while its baseline is worse. Scored against the same seed's
baseline in sibling cells the pair would read -8.4% (map_xattn_pe_fix) or -7.9%
(map_shufflepose_fix). So roughly half the extremity is the bad denominator and half is a
genuinely strong oracle run; it is not purely an artifact.

ACTION: queued one resample of MS_map_nopose_fix_s404 (tools/scripts/run_baseline_s404_resample.sh,
waits for the main queue so the two never race the GPU). This is NOT cherry-picking, and the
ordering is the reason: the outlier criterion -- a >2 SD within-cell baseline from a seed that is
not systematically slow -- was committed in fb8eee7 hours BEFORE this oracle run finished. The
original is kept, not overwritten; detection_floor.py now scores every rung under each available
denominator (all seeds / without s404 / s404 resampled) and shouts if the bracket edge moves
between them, in which case the edge is reported as unresolved rather than picked. If the resample
also lands near 0.495 then that is simply the seed's value and the original stands.

detection_floor.py was restructured for this: per-rung statistics factored into stats_for(), rows
rebuilt per denominator by build(), and infer_bracket() lifted to module scope (exercised on
monotone / non-monotone / all-sig / none-sig / unsorted). The printed table still reproduces the
published numbers exactly.

### 2026-10-23 paper numbers checked against artifacts; one rounding error fixed

Mechanically compared every Sec. IV-B number against experiments/facing_error_ladder_units.json.
Nineteen of twenty matched. One did not: the box heading's +3 s median is 3.884 deg, which rounds
to 3.9, and I had written 3.8 in two places in the paper and in detection_floor.py's reference
constant. Fixed. Also made the movers share exact (95.5%, was written as 95%).

Saved the check as tools/scripts/verify_paper_numbers.py, because this paper keeps accumulating
derived numbers and this class of error has now happened three times in one day: 44.2 deg quoted
against a floor calibrated on a different quantity, a rel column printed with a different
estimator from the one tested, and now a mis-rounded median. The script rounds each artifact value
the way the paper writes it and asserts the string is present, so it catches the paper and the
artifact drifting apart in either direction. It also surfaces detection_floor.json's new
bracket_edge_stable flag, so a bracket that is not stable across denominators cannot be quoted as
a single edge.

It currently reports one mismatch, correctly: rung 10.2's MDE is 12.7 at n=4 while Table II was
built at n=3 (6.2). That is Part A in flight, not an error -- the rung's n is printed in the label
so an in-flight divergence reads as "refresh the table when the queue lands". Table II and the
bracket both need regenerating once Part A finishes.

Paper rebuilds clean, 8 pages, no undefined references.

### 2026-10-23 PART A complete at n=5: the bracket is UNRESOLVED, and one baseline decides it

All four runs landed (15:11). The bracket edge moves depending on a single denominator:

  rung    n=5 all seeds                     n=4 without s404
  10.2    -6.41%  p=0.054  MDE 8.8          -4.14%  significant  MDE 3.8
  14.6    -1.86%  p=0.447  MDE 8.2          +0.24%  n.s.         MDE 3.7
  bracket [5.5, 10.2]                       [10.2, 14.6]

THE ANOMALY IS THE SHARED DENOMINATOR, not the oracle runs. This is now established
non-circularly, from the oracle runs' agreement with each other rather than from the disputed
relative differences:

  rung   s404 oracle   others avg   oracle gap   s404 rel   others rel
  10.2      0.4188       0.4324       -3.13%      -15.45%     -4.14%
  14.6      0.4447       0.4521       -1.64%      -10.24%     +0.24%

At rung 14.6 the s404 oracle run is 1.6% better than the other four -- unremarkable -- yet reads
-10.24%. Its baseline (0.4954) is +9.8% above the other four (0.4511, range 0.4428-0.4584), and
that factor divides into every s404 arm identically. So the same seed reads extreme at BOTH rungs
in the same direction while its numerators are ordinary, which is the signature of a bad common
denominator and not of two lucky oracle runs.

PART A THEREFORE DID NOT CONVERT THE BRACKET INTO A MEASUREMENT. It established something more
useful: the bracket's location was resting on one baseline run all along, and at n=3 that was
invisible. The queued resample of MS_map_nopose_fix_s404 now decides between [5.5, 10.2] and
[10.2, 14.6]. It runs after the main queue, so ~23:00-24:00.

PAPER CONSEQUENCES IF [10.2, 14.6] STANDS -- not applied yet, the resample decides:
- The abstract's "already indistinguishable from zero at 10.2 deg" becomes false; at n=5 it is
  p=0.054 all-seeds and significant without s404. Even the all-seeds reading is now marginal
  rather than a clean null, so that sentence needs softening either way.
- Pooled SMPL facing (12.3 deg) would sit INSIDE the bracket. The paper survives, but Sec. IV-B's
  three-way decomposition (heavy tail with 56.8% of agents outside; the median is trajectory-
  derived via root-orientation snapping, with the stationary subset at 41.2 deg; the box heading
  is a better copy at 3.9 deg and already an input) stops being a supporting argument and becomes
  THE argument. That decomposition is already written, which is lucky.
- Table II needs regenerating with mixed n and the per-rung MDE column, and the bracket must be
  quoted both ways until the denominator is settled.

Note on method: both of the guards added earlier today fired on real conditions within hours --
the completion gate (which caught a 2/30-epoch run scoring as finished) and the bracket-edge
stability check. Neither was hypothetical.

### 2026-10-23 PART B rung 0 at n=2: early fusion exploits the oracle on one seed, not the other

8/13 runs done. Rung 0 (uncorrupted direction oracle) is scorable at n=2 and the two seeds
disagree by 15 pp:

  early rung 0.0   s101  0.3942  -14.01%      late rung 0.0   s101 -16.47%
                   s202  0.4460   +0.72%                      s202 -19.33%
                                                              s303 -11.48%  (sd 3.97 pp)

The channels are bit-identical between pathways at a given rung (verified earlier today), and
late fusion converted them on all three seeds at -11 to -19%. So s202 is not an information
problem; the early-fusion run simply failed to exploit a cue that was present in its input.

Its trajectory says the same: the last six evaluated epochs are 0.4738 0.4731 0.4829 0.4887
0.5244 0.4460 -- drifting worse, then an 0.08 jump at epoch 30, with "best" being the final
epoch. s101 by contrast sat stably at ~0.395. So s202 is parked near the no-pose level (its
baseline is 0.4428) and was arguably still descending when training stopped.

This is NOT a general early-fusion instability. Last-6 spread across runs: early 0.0156 mean
(0.0070, 0.0114, 0.0182 and s202's 0.0256), late 0.0115 (0.0082-0.0164). Only s202 stands out,
and it stands out in LEVEL more than in spread.

CONSEQUENCE FOR PART B. If rung 0's spread is ~10 pp at n=3, the zero-crossing cannot be located
and Part B comes back uninterpretable -- the floor is read off where the decay curve crosses zero,
which needs the rungs to be separable. s303 at rung 0 (two runs out) decides whether s202 or s101
is the outlier. Not queuing anything extra: the remaining 5 runs will show whether this resolves,
and the options then are more seeds, more epochs, or reporting Part B as inconclusive. That is the
user's call.

Worth noting the single-seed shape from s101 pointed the same way earlier: early fusion matched
late fusion at rung 0 but had a 2.6 deg half-life against late's 5.1 deg. If early fusion both
exploits the cue less reliably AND loses it faster, the pathway the paper recommends has a
STRICTER floor than the one it measured, which strengthens the paper's conclusion rather than
weakening it. Too early to assert.

### 2026-10-23 PART B: seed 202 fails to exploit early fusion at BOTH rungs

9/13. Rung 5.5 is now scorable and s202 repeats its rung-0 behaviour, so the earlier entry's
"s303 decides whether s202 or s101 is the outlier" now has a sharper framing: this is a seed x
pathway interaction, not one bad run.

            early           late (same seed)
  rung 0.0  s101 -14.01%    s101 -16.47%
            s202  +0.72%    s202 -19.33%
  rung 5.5  s101  -3.20%    s101  -7.74%
            s202  +1.16%    s202  -7.25%

Seed 202 converts the cue strongly through the side branch (-19.3%, -7.3%) and not at all through
the agent channels (+0.7%, +1.2%), with the SAME bit-identical input tensor. Seed 101 converts it
through both. So the information is there and the failure is in optimisation.

The trajectories agree. s202's early runs both drift worse through epochs 25-29 and then jump at
epoch 30, with "best" being the final epoch:
  rung 00  0.4738 0.4731 0.4829 0.4887 0.5244 0.4460   sd 0.0256
  rung 05  0.4576 0.4811 0.5192 0.5148 0.5343 0.4627   sd 0.0321
against s101's early runs at sd 0.0070 / 0.0114 / 0.0182. Late-training divergence specific to
that initialisation with six extra input channels, not a general early-fusion instability.

Running tallies: 2/2 s202 early runs failed to exploit, 3/3 s101 succeeded, 2/2 s202 late runs
succeeded. n=2 statistics are meaningless as expected (rung 0 MDE 103.7, rung 5.5 MDE 30.7).

WHAT THIS LIKELY MEANS FOR THE PAPER -- stronger than the floor measurement it replaces. Part B
was asked "what is the floor of the pathway we recommend?" The answer forming is "that pathway
does not reliably use the cue at all": early fusion is bimodal across initialisations, succeeding
on one seed and failing on another with identical input. For a paper that recommends early fusion
on the grounds that it carries no capacity tax, an unreliability result is a more important caveat
than a slightly different floor, and it is a practical warning rather than a null.

Still not queuing anything. s303 (two runs out) tells us whether the split is 2-1 or 1-2, and the
honest options then are: report the bimodality with n=3 and drop the early floor claim; add seeds
to characterise the failure rate; or investigate the late-training divergence (LR schedule, or
gradient scale on the appended channels). That is the user's call.

### 2026-10-23 PART B: s202 fails at all three early rungs, s101 succeeds at all three

10/13. Rung 10.2 completes the 2-seed picture and the split is clean:

    rung   s101 early  s202 early  |  s101 late  s202 late
    0.0d     -14.01%      +0.72%   |   -16.47%    -19.33%
    5.5d      -3.20%      +1.16%   |    -7.74%     -7.25%
   10.2d      -3.55%      +2.87%   |    -2.39%     -2.76%

3/3 s202 early runs fail to exploit, 3/3 s202 late runs succeed, 3/3 s101 early runs succeed.
All three s202 early runs carry the same trajectory signature -- drift worse through epochs 25-29,
jump at epoch 30, best = final epoch -- with the spread growing with rung:
  rung 00  0.4738 0.4731 0.4829 0.4887 0.5244 0.4460   sd 0.0256
  rung 05  0.4576 0.4811 0.5192 0.5148 0.5343 0.4627   sd 0.0321
  rung 10  0.4629 0.4651 0.5208 0.5015 0.5465 0.4596   sd 0.0361

The within-seed consistency matters: this is not random per-run failure but a reproducible
property of that initialisation combined with the early-fusion architecture. (The seed sets init
and data order; the oracle's angular noise is deterministic by crc32 of scene/sigma/object and so
is identical across seeds at a given rung.) Three independent runs of the same seed failing the
same way, while the same seed succeeds three times through the other pathway on the same input,
is a stronger statement than three unrelated bad runs would be.

n=2 statistics remain meaningless (MDE 103.7 / 30.7 / 45.2); the per-seed values are the content.
s303 decides whether the split is 2-1 or 1-2.

### 2026-10-23 PART B rung 0 at n=3 -- early fusion leaves a PERFECT oracle inert in 2 of 3 seeds

11/13. s303 also fails to exploit, so the split is 1-2, not 2-1.

  seed   early oracle   early rel   late rel   last-6 sd
  s101        0.3942     -14.01%    -16.47%      0.0070   exploited
  s202        0.4460      +0.72%    -19.33%      0.0256   INERT
  s303        0.4516      -0.31%    -11.48%      0.0161   INERT

  early rung 0: mean  -4.53%  sd 8.22 pp  p=0.440  MDE 25.5
  late  rung 0: mean -15.76%  sd 3.97 pp  p=0.021  MDE 12.3

Rung 0 is a PERFECT direction oracle -- the agent's own future bearing, uncorrupted. It is the
best case the pathway will ever see. Late fusion converts it on 3/3 seeds (-11.5 to -19.3%).
Early fusion converts it on 1/3, from a bit-identical input tensor. Epoch trajectories: s101
settles at ~0.395, while s202 and s303 wander at 0.47-0.52 and never approach it.

THIS PROBABLY REINTERPRETS A RESULT THE PAPER ALREADY REPORTS. Sec. III-B finds early fusion
carries no branch-capacity tax: it performs the same as no-pose. Two explanations fit that:
  (a) early fusion integrates pose at no capacity cost          <- the paper's reading, favourable
  (b) early fusion largely ignores the appended channels         <- nothing to cost, unfavourable
The oracle discriminates, because (a) predicts a large gain from a perfect channel and (b)
predicts nothing. We see nothing in 2 of 3 seeds. So "no tax, no benefit" is at least partly the
signature of an inert input, which is not a virtue, and the paper currently RECOMMENDS this
pathway on the strength of it.

Limits, stated plainly: n=3; and the paper's early-fusion cells (fa_derived, fa_derived_shuf) ran
at 25k scenes and LR 5e-5 while these run at 579 and LR 1e-4, so the reinterpretation is
suggestive at this scale and not demonstrated at theirs. Testing it where the claim lives would
mean an oracle arm at 25k.

CONSEQUENCES:
- Part B cannot measure an early-fusion floor. A floor is where a decay curve crosses zero, and
  there is no reliable curve: 2 of 3 seeds sit at zero from rung 0 onward. The honest Part B
  result is the unreliability, not a bracket.
- Sec. III-B's "early fusion removes the tax" needs the inertness alternative stated, and the
  recommendation to prefer early fusion needs withdrawing or heavily qualifying.
- This strengthens rather than weakens the paper's thesis: both integration pathways fail, for
  different and now-identified reasons -- the side branch pays a capacity tax and has a detection
  floor above SMPL's accuracy, while early fusion avoids the tax by largely not using the channel.

Two runs left (earlyoracle05/10_s303), then the s404 baseline resample.

### 2026-10-23 PART B rung 5.5 at n=3 -- the pathway contrast is now stark

12/13.

  rung 0.0d  early s101:-14.01%  s202:+0.72%  s303:-0.31%   mean -4.53%  sd 8.22
             late  s101:-16.47%  s202:-19.33%  s303:-11.48%  mean -15.76%  sd 3.97
  rung 5.5d  early s101: -3.20%  s202:+1.16%  s303:-0.82%   mean  -0.95%  sd 2.18  MDE 6.8
             late  s101: -7.74%  s202: -7.25%  s303: -7.69%  mean  -7.56%  sd 0.27  p=0.0004

Two things stand out. The late ladder at rung 5.5 is almost noiseless -- three seeds within 0.5 pp
of each other, sd 0.27 -- which is why that rung's MDE is 0.8 and why the published floor has a
sharp lower edge at all. And even the one seed where early fusion works loses the cue far faster:
s101 goes -14.01% -> -3.20% between rungs 0 and 5.5, while late fusion holds -7.6% across all
three seeds at 5.5.

So the early pathway is both unreliable (1 of 3 seeds exploits a perfect oracle) and, where it
does work, shorter-ranged. Nothing here changes the conclusion already logged; it sharpens it.
One run left (earlyoracle10_s303), then the s404 resample.

### 2026-10-23 23:02 QUEUE FINISHED (13/13, no failures). PART B complete.

  rung | early per seed                         mean     sd      p  | late mean    sd       p
  0.0d | s101:-14.01  s202:+0.72  s303:-0.31  -4.53%   8.22  0.440 |  -15.76%  3.97  0.0205
  5.5d | s101: -3.20  s202:+1.16  s303:-0.82  -0.95%   2.18  0.528 |   -7.56%  0.27  0.0004
 10.2d | s101: -3.55  s202:+2.87  s303:+1.16  +0.16%   3.32  0.941 |   -6.41%  5.30  0.0540

PART B RESULT: no rung is significant, including rung 0. There is no decay curve to find a
zero-crossing on, because there is no reliable gain to decay. The early-fusion pathway has no
measurable detection floor for want of a measurable gain -- 1 of 3 seeds exploits an uncorrupted
oracle, against 3 of 3 for the side branch, from a bit-identical input tensor.

This answers the question Sec. IV-B raised ("we have not measured the floor of that cheaper
pathway") with something other than a number, and the honest write-up is that the pathway does not
reliably convert the cue at all. See the 21:xx entry for why that probably also reinterprets
Sec. III-B's "early fusion removes the branch-capacity tax": a channel that is largely ignored
costs nothing and buys nothing, which is what both results look like together.

The resample handoff worked: the waiter saw the queue finish at 23:02:46 and started
MS_map_nopose_fix_s404_resample at 23:03:06. That run decides the Part A bracket.

STILL OPEN AT THIS POINT:
1. s404 baseline resample -> Part A bracket is [5.5, 10.2] or [10.2, 14.6]. ~35 min out.
2. Paper edits, none applied yet: Table II needs mixed-n regeneration; the abstract's
   "indistinguishable from zero at 10.2 deg" is marginal at best (p=0.054 at n=5); Sec. III-B
   needs the inertness alternative and the early-fusion recommendation qualified; Sec. IV-B's
   "we have not measured the floor of that cheaper pathway" can now be answered.
3. Page budget: 8 pages against a 6-page limit, cuts still deferred.

### 2026-10-23 paper updated for Part B; back to 8 pages (the actual limit)

Page limit confirmed as 8 including references, so no content needed cutting for budget -- but
adding Part B pushed it to 9, with page 9 holding only references, so about one column had to come
back out.

APPLIED:
- New Sec. IV-C "The cheaper pathway has no floor to measure": the early-fusion ladder, its
  per-seed values, and the argument that this is the absence of a curve rather than a lower floor.
  Explicitly withdraws the paper's earlier suggestion that a cue failing through the branch might
  still pay if fused early.
- Sec. III-B's free-carriage sentence now flags the inertness alternative ("a channel it costs
  nothing to carry is also a channel it largely does not read").
- Architecture figure caption, the practice section, and the limitations all updated: the paper no
  longer says the floor belongs only to a late-fused branch, because we laddered two pathways.
- TWO LEFTOVERS OF THE RETRACTED "FACTOR OF FOUR" CLAIM, found by grep and fixed. The general
  condition said pose fails "by about a factor of four in bearing" and the conclusion said facing
  is "roughly four times coarser than that bracket". Both were the mixed-units comparison removed
  from Sec. IV-B this morning and both survived that edit. Now: outside the bracket for most
  pedestrians, and a factor of four only for the stationary ones, where 41.2/10.2 really is 4.0.
  Worth noting the paper-number checker did not catch these -- it verifies quantities present in
  the artifacts, not retracted claims restated in prose. A grep for the retracted phrasing is the
  thing that caught them, which is a lesson about what mechanical checking does and does not buy.
- Conclusion gains one sentence on both pathways failing for identifiable and different reasons.

TRIMMED back to 8 pages, ~280 words, all redundancy rather than substance: the early-fusion table
(its per-seed values were already the prose's content); the onset paragraph's restatement of what
the onset task is; a duplicated "measure it on the pathway you will ship" in the practice section;
the "not a factor of four" rebuttal in Sec. IV-B, which only made sense against a draft no reader
will see. Every measurement in the paper survived. Builds clean, one 2.6pt overfull box.

STILL PENDING, both waiting on the resample (epoch 17/30):
- Table II regeneration at mixed n. verify_paper_numbers.py correctly flags rung 10.2's MDE as
  8.8 at n=5 against the table's 6.2 at n=3.
- The abstract's "already indistinguishable from zero at 10.2 deg", which is p=0.054 at n=5
  all-seeds and significant without s404.

### 2026-10-24 00:xx RESAMPLE IN: the floor bracket moves to [10.2, 14.6]

MS_map_nopose_fix_s404_resample = 0.4729, against the original 0.4954. The pre-specified test
(criterion committed in fb8eee7, hours before the s404 oracle runs finished) came back positive:
the original draw was anomalous.

  denominator for s404        rung 10.2        rung 14.6        bracket
  original 0.4954 (as pub.)   -6.41% p=0.054   -1.86% p=0.447   [ 5.5, 10.2]
  resampled 0.4729            -5.60% p=0.026   -1.00% p=0.519   [10.2, 14.6]
  mean of both 0.4842         -6.01% p=0.040   -1.44% p=0.472   [10.2, 14.6]
  dropped (n=4)               -4.14% p=0.020   +0.24% p=0.804   [10.2, 14.6]

Three of four treatments give [10.2, 14.6]; only keeping the single original draw gives
[5.5, 10.2]. Both draws are legitimate samples, so the MEAN is now the primary denominator in
detection_floor.py and the single-draw readings are the sensitivity analysis.

AN UNCOMFORTABLE SECOND FINDING. The resample is still +3.36 sigma above the other four baselines
(the original was +6.82), and the two draws of the SAME seed differ by 0.0225 -- more than the
other four baselines' entire range (0.0156). So run-to-run noise at fixed seed rivals the
between-seed spread. The paired t-test is still valid (it uses the observed spread of the per-seed
differences, and never assumes a noiseless baseline), but it means a large share of the ladder's
variance is baseline noise, which is why the MDEs are what they are. Averaging baseline draws
would buy more power per GPU-hour than adding seeds. Noted, not acted on.

CONSEQUENCES, ALL APPLIED TO THE PAPER:
- Bracket [5.5, 10.2] -> [10.2, 14.6] in the abstract, intro, Table II caption, Sec. IV, the
  point-cloud caption and the conclusion.
- Eq. (1) refit: -16%*exp(-theta/8.1 deg), half-life ~6 deg (was 6.4 deg / ~4 deg). The per-seed
  constants (amplitudes 12-20%, half-lives 3.3-8.1 deg) are UNCHANGED, because they are fit on
  seeds 101/202/303 only and never touch the s404 denominator -- so Sec. IV-B's per-seed
  predictions (1.5-4.1% at 12.3 deg, 0.003-0.35% at 41.2 deg) also stand as published.
- Table II gains an n column (3, or 5 at the two bracket rungs) and the two rungs are updated.
- A new paragraph states the denominator dependence plainly and calls the lower edge the least
  robust number in the paper.
- THE HEADLINE CHANGES CHARACTER. Pooled SMPL facing (12.3 deg) now falls INSIDE the bracket, so
  "pose is below the floor" is no longer true of the pooled median. Sec. IV-B's three-way
  decomposition is now the whole argument rather than a reinforcement: roughly half of pedestrians
  sit above the bracket (43.5% above 14.6 deg at +3 s, 51.1% at +8 s); the pooled median is
  inherited from root-orientation snapping, and the stationary subset the cue was meant to serve
  sits at 41.2 deg; and the box heading is a better copy (3.9 deg, above the bracket for only
  14.9% of agents) that the model already receives. That paragraph was written this morning for
  exactly this contingency, which is the only reason this landed as an edit rather than a rewrite.
- The abstract's "the null is not a near miss" is GONE. In pooled median terms it is nearer than
  that sentence claimed, and the honest framing is the decomposition.
- Fractions updated throughout: the threshold that matters is now the bracket's upper edge
  (14.6 deg), not 10.2.

Back to 8 pages: the denominator paragraph was tightened, two paragraphs trimmed, and the two
figures scaled to 0.80\textwidth and 0.94\linewidth. No measurement was cut. Zero overfull boxes,
no undefined references, all 17 references on page 8, and verify_paper_numbers.py passes 30/30 --
after fixing a false pass in the checker itself, which was validating the 'all seeds' bracket
while the paper quotes the primary one (5.5 and 10.2 still appear as rung labels, so it passed
without checking anything).

### 2026-10-25 point-cloud caption contradicted the paper's own per-joint table

The fig:pcmesh caption said the mesh's articulation "and any gait or facing cue read from it,
comes largely from the AMASS motion prior rather than from the measurement". Stated generally
that is the retracted "gait at the noise floor" argument leaking back in through a caption:
Sec. V's own Table (n=534 pedestrians, 1,789 GT-keypoint frames) scores every limb joint at
+0.46 to +0.64 skill, so limbs ARE measured. The only joint the pipeline emits from its prior is
the HEAD, at skill 0.05 [0.02, 0.07] -- MPJPE 112 mm against a GT spread of 117 mm.

Caption now scopes the prior-dominance claim to the far panels it actually depicts (12-38
returns), points at Table~\ref{tab:skill} for what is and is not recovered, and demotes "this is
the physical reason body facing falls above the bracket" to range-sparsity being ONE contributor,
cited against the evidence that does support it (the pose-versus-heading gap widening to
+32.9 deg at 41-75 m). Still 8 pages, no overfull boxes, 30/30 number checks.

Surfaced while assessing a proposal to turn WOD Perception into a prediction dataset. Recorded
there because it changes which cue is open: gait is measured and was tested (null), facing looks
intrinsically limited (the box heading, far better measured, still only reaches 25.2 deg on onset
direction, outside the [10.2, 14.6] bracket), and HEAD ORIENTATION is the one channel the LiDAR
pipeline demonstrably fails to deliver. That is the argument for cameras, and it is why the WOD
question is worth a diagnostic rather than a dataset build -- see [[wod-prediction-dataset]]
reasoning in the session: the ceiling test needs GT pose at ONE instant plus the tracked future,
not 9.1 s of keypoint continuity, so it costs no training at all. Feasibility gate before any
build: count stationary pedestrians with GT keypoints and a usable future; 534 locally implies
~25 stationary, which is too few.

### 2026-10-25 retired-claim grep added to verify_paper_numbers.py

The number checker verifies quantities that exist in the artifacts; it cannot catch a claim the
project has WITHDRAWN being restated in prose. That happened twice in three days -- "a factor of
four in bearing" survived the Sec. IV-B units fix and reached the conclusion, and "gait at the
noise floor" re-entered through a figure caption after the per-joint table had retired it. Both
were caught by grep, so the grep is now part of the checker.

Eight entries, each with why it was retired and what to say instead, and an optional `allow`
regex for occurrences that are legitimate in their correct scope: 44.2 deg is fine next to
"onset", [5.5 deg bracket is fine next to "original draw", and "might still pay if fused early"
is fine next to "withdraw" because the paper states it in order to withdraw it. That last one
fired on the first run, which is the design working.

Negative-tested. The FIRST negative test silently verified nothing -- the sed pattern I injected
with never matched, so the scratch file was byte-identical to the paper and the checker correctly
passed. Exactly the failure the checker exists to prevent, in the test for the checker. Injecting
on a phrase that does occur gives exit 1 with the right line and guidance. Added a PAPER_TEX env
override so this can be tested against a scratch copy without touching the real paper.

Caveat worth keeping in mind: RETIRED is a manual list. It catches regressions of claims someone
remembered to add, not claims nobody wrote down.

### 2026-10-26 the 579 rung: clean data, but the tax comparison is cross-scale

Raised as "579 is weird, it came from the extraction bug — replace it with 5k or 25k". Two
corrections and one real finding.

1. THE 579 DATA IS CLEAN. `final_10fps_subset579` was created 2026-08-19 01:49, three minutes
   after the corrected 299,220-scene re-export, so it is a subset carved from the good export.
   The bug explains why the rung is 579 scenes (that was all that existed when the ladder was
   first built) but not the quality of what is in it.

2. THE REAL PROBLEM IS A SCALE MISMATCH, not the number. Sec. IV-B compares the ladder's
   predicted gains (579 rung) against the ~1.6% branch tax (25k rung). Measured at both:
     579 rung  n=5  +0.61%  p=0.800  CI [-5.64,+6.85]
     25k rung  n=5  +1.67%  p=0.001  CI [+1.22,+2.12]
   At 579 the tax is unresolvable at +/-6 pp, so it CANNOT be quoted at the ladder's own scale.
   The paper now discloses this in one clause rather than pairing the two silently.

3. MIGRATION COST, from measured wall times (50 min/run at 579, 17.0 h/run at 25k):
     rung   per run   full 8-rung ladder n=3   bracket rungs (05/10/15) n=3
     579     0.8 h     20 h  (0.8 d)            7.5 h (0.3 d)
     5k      3.4 h     82 h  (3.4 d)           31   h (1.3 d)
     25k    17.0 h    409 h (17.0 d)          153   h (6.4 d)
   No-pose baselines already exist at 25k (scale25k_nopose, n=5) and 579; 5k has none, so a 5k
   ladder costs 3 extra baseline runs.

RECOMMENDATION: do not move the whole ladder. The full 8-rung ladder at 25k is 17 days of
single-GPU compute and most of its rungs (27-57 deg) only establish that the gain is gone, which
579 already shows. If the scale mismatch is worth GPU, the targeted buy is rungs 05/10/15 at 25k,
n=3 -- 6.4 days, baselines already in hand -- which would state the floor and the tax at one
scale and incidentally retire 579 from the headline. Everything below the bracket (rung 0's
-15.8%) is far larger than any plausible tax, so those rungs do not need re-running at scale.
