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
