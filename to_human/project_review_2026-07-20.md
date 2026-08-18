# Full Project Review — 2026-07-20

Scope: paper (`paper/iclr2026/main.tex`), presentation (`presentation/MTR-Pose_presentation.pptx`),
analysis (`experiments/multiseed_analysis.json` + `tools/scripts/analyze_multiseed.py`), code, docs.
All numeric claims re-verified against a fresh, byte-identical regeneration of the analysis JSON.

---

## Part 1 — Issues found & fixed (commits `0d12229`, `33d0f42`, `bf62dd1`)

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | `make_figures.py` plotted xattn_pe with **3 seeds** in Fig 2 (scatter) and Fig 3 (training bands) — figures visually supported the retracted "stable −3.5%" claim | serious | switched to 5 seeds, regenerated; Fig 3 now shows the seed-505 failure band |
| 2 | Appendix text: "xattn+PE band is the narrowest of all variants" — false at 5 seeds | serious | rewritten; names the 3-seed tightness as the sampling artifact |
| 3 | Reproducibility note claimed encoder/map/pretrain cells are "mask-insensitive" — false (they carry active WTA-L1; wta01 unmasked 0.6853 vs masked 0.6715) | serious | honest scope: pre-fix code for all tabled results; only baseline verified + audit cells re-run masked |
| 4 | ~30 uncorrected pairwise tests, unacknowledged | moderate | Limitations: p≤1e-4 effects survive Bonferroni at this family size; 0.01–0.05 range labeled exploratory |
| 5 | `baseline→baseline_masked` p=0.95 was the *paired* p (hier = 0.63), unlabeled | minor | labeled both |
| 6 | Failure-rate claim "5/25 (20%)" stale after new seeds | minor | now 6/27 (22%, minADE ≥ 0.70) |
| 7 | Pretrain table caption said "three seeds per cell" (three rows are n=5) | minor | corrected |
| 8 | Deck slide 8 still had "(Insert figure…)" note | minor | removed via python-pptx |

**Verified clean:** all 19 cell means/stds + key pair CIs/p-values in paper & deck match the JSON;
pytest 4/4; paper 9 main pages, 0 undefined refs; JSON identical to fresh regeneration.

---

## Part 2 — Missing experiments (ranked) & tracking

- [x] **2.1 Root-orientation ablation in the MAP condition** — DONE 2026-07-21. RESULT: map_norootorient 0.4800±0.0046 ≈ map_xattn_pe (p=0.58), still −1.55% vs no-pose+map → the map-condition benefit is NOT heading re-entry; mechanistic claim reversed in the paper (favorably).
  The mechanistic headline ("even the surviving −1.6% is mostly heading re-entry") is extrapolated
  from the no-map ablation to the map×xattn+PE cell where it was never tested.
  Plan: `mtr+pose_data_cross_attn_pe_with_map_norootorient.yaml`, 3 seeds (101/202/303),
  tag `MS_map_norootorient_s<seed>`. Decision rule: if map_norootorient ≈ map_nopose, the heading
  story extends to the map condition; if map_norootorient ≈ map_xattn_pe, the map-condition gain is
  NOT heading-driven and §4.4/§4.5/Limitations/Conclusion must be softened.
- [x] **2.2 EVERYTHING to n=5** — DONE 2026-07-21 (34 runs; pre-fix cells via worktree per Part 4). Key: clean xattn_pe collapse replicates (.710/.703); masked '+5.5%' claim dead; pretrain pose n.s.; 30fps null firm.
- [~] **2.3 External metrics anchor** — MR@2m implemented in analyze_multiseed.py (fraction of
  evaluated pedestrians with best-epoch minFDE > 2 m; NOT the official velocity-gated Waymo
  MissRate). Sanity: baseline 0.239, map 0.145, pretrained 0.093. Paper integration after the
  run queue finalizes numbers. **Official Waymo mAP: BLOCKED** — tensorflow + waymo_open_dataset
  not installed in mtr_smpl, AND result.pkl (full predictions) were deleted by the matrix
  scripts with only last-epoch checkpoints kept, so best-epoch mAP is impossible for existing
  runs; DECIDED (Erik, 2026-07-20): option (b) —
  drop official mAP; keep MR@2m as the external anchor and disclose in the paper why official
  Waymo metrics are not reported (predictions not retained; only last-epoch checkpoints kept).
  Paper text lands together with the final MR@2m numbers after the run queue completes.
- [x] **2.4 World-frame pose confound** — IMPLEMENTED 2026-07-20: `rotate_root_6d` helper +
  `AGENT_CENTRIC_POSE_ROT` dataset flag (root joint 0 rotated by Rz(-heading), matching the
  trajectory transform; body joints parent-relative, untouched). Unit-tested (3 tests).
  Config `mtr+pose_data_cross_attn_pe_with_map_agentrot.yaml`; 3 seeds (`MS_map_agentrot_s*`)
  chained as phase5d behind the review queue. Pairs registered in analyze_multiseed.py.
- [ ] **2.5 Pose prediction quality eval** — predictions discarded at eval time. *(Erik will look into this.)*

---

## Part 3 — Academic-correctness assessment (post-fixes)

**Solid:** single source of truth for all numbers (regenerable JSON); inert-loss claim is analytical +
numerically verified; double self-correction documented, not hidden; conservative hierarchical test
used consistently; multiplicity acknowledged; reproducibility scope honest; 30 fps control replicated.

**Known soft spots (disclosed, not hidden):**
1. Hierarchical bootstrap with 3–5 seeds is coarse (3 seeds ⇒ only 10 distinct multisets); p-values
   0.05–0.15 from 3-seed cells carry limited meaning. Possible add-on: seed-level permutation test.
2. Best-checkpoint selection inflates all "best" values (last-5-mean is in the JSON, not the tables).
3. Single validation split; 579-scene training set; the positive result is one configuration on one
   small benchmark — the generalizable contribution is the methodology.
4. `research-log.md` has an uncommitted pre-existing edit (not from this workstream — Erik to review);
   `to_human/opus48_action_plan.md` untracked by choice.

---

## Part 4 — CODE-VERSION MIXING FLAW (found 2026-07-20, after the review above)

Planning the "everything to n=5" extension exposed a provenance flaw the review missed:
the P1.1 mask fix (commit `d987de2`, 2026-07-08) changed the training distribution of
every aux-active cell, and runs before/after it are NOT interchangeable.

- **Pre-fix runs (2026-06-12 matrix):** baseline(5), geo_pure(5), wta01(5), gmm_only(3),
  mpjpe(3), full(3), xattn(3), xattn_pe **101/202/303**, map_nopose/map_wta01/ft_nopose/
  ft_wta01 101/202/303.
- **Post-fix (masked) runs:** everything from Phase 4 on — including **xattn_pe 404/505**.

**Consequence:** the paper's 5-seed xattn_pe cell mixed 3 pre-fix + 2 masked runs; the
seed-505 "failure" that drove the collapse retraction came from a different training
distribution (masked ⇒ WTA aux inert). The retraction *direction* may still hold, but the
cell was not clean. Cells where a naive 404/505 extension would repeat the mistake:
map_wta01, xattn, gmm_only, mpjpe, full, ft_wta01.

**Resolution (in progress):**
1. Mixed runs renamed: `MS_xattn_pe_s{404,505}` → `MS_xattn_pe_maskedcode_s{404,505}`
   (preserved; registered as their own cell — effectively "xattn+PE, no active aux").
2. Git worktree pinned to the pre-fix Phase-0 commit (`70c7a5d`) at
   `/home/erik/ssd2/gitprojects/MTR_prefix_worktree` (data + CUDA .so symlinked).
3. Serial run queue (`run_review_queue.sh`): phase5 (map_norootorient×3, norootorient
   404/505 — current-code-consistent) → phase5b (pose30fps + 5 masked cells 404/505 —
   current-code-consistent) → phase5c (**worktree, pre-fix code**: xattn_pe, xattn,
   gmm_only, mpjpe, full, map_wta01, ft_wta01 404/505).
4. RESOLVED 2026-07-21: pre-fix 404/505 = 0.7098/0.7029 — the collapse REPLICATES under
   clean provenance; the retraction stands (stronger). Mixed-cell flaw did not drive the
   conclusion, but the record is now clean.
5. Note for mixed-comparator pairs: map_nopose/ft_nopose 404/505 (phase4e) are post-fix,
   but those cells are no-pose; the fix was verified to leave the no-pose baseline
   unchanged (p=0.95), so treating them as one cell is defensible — disclosed here.

## Part 5 — Professor feedback follow-ups (2026-08-12)

Four ToDos from Erik's advisor. All target the ONE surviving result (map x xattn+PE,
-1.57%, p<1e-4). Scope confirmed with Erik: #4 is a representation ablation inside our
framework, NOT a head-to-head against a published skeleton method (separate codebase,
different datasets/metrics -> cited, not benchmarked).

- [x] **F1 Wrong-pose control — COMPLETE, RESULT OVERTURNS THE POSITIVE CLAIM** (`SHUFFLE_POSES_ACROSS_AGENTS`). Each pedestrian gets
  another pedestrian's pose sequence (fixed seeded derangement, 0 self-assignments).
  Donor *content* is placed on the *recipient's* observation grid, so the validity masks
  are bit-identical (verified 60/60 scenes; 1567 == 1567 valid pose frames) -- otherwise
  the shuffle silently dropped 28% of valid pose frames and "wrong pose" would have been
  confounded with "less pose". Decision rule: if the -1.6% survives wrong poses, the
  benefit is capacity/regularization, not pose information. 5 seeds, phase6.
  **RESULT (2026-08-12): map_shufflepose = 0.4797 +- 0.0038 (n=5) = -1.18% vs no-pose
  (paired p=1e-4, hier p=0.05). Correct vs WRONG poses: +0.40%, hier p=0.29 -> NOT
  distinguishable.** Wrong poses recover ~75% of the -1.57% benefit. Every pose-branch
  variant clusters at 0.478-0.481 regardless of content (correct .4778 / root-zeroed
  .4800 / agent-frame .4801 / WRONG .4797 / keypoints .4807 / GRU .4823) vs no-pose
  .4854. => The gain is the pose BRANCH (capacity/optimization), not pose CONTENT.
  This also re-reads the 2.1/2.4 controls: root-zeroing and frame-change looked harmless
  not because body configuration carries the signal, but because content barely matters.
- [~] **F2 Two-second horizon.** Both arms retrained at NUM_FUTURE_FRAMES=20 with
  intention points RE-CLUSTERED from 2s endpoints (shipped goals span 1.7-49.2 m; true
  2s endpoints span 0.02-7.48 m -- reusing them would mis-specify the goal head).
  Plus horizon-decomposed minADE@2s/4s/8s now recorded per agent. 10 runs, phase7.
- [x] **F3 Subgroup selection — COMPLETE, negative result.** Discovery half A / confirmation
  half B (hash split); pose-independent descriptors in two families (`past_*` deployable,
  `future_*` oracle); full heuristic x threshold scan on A only; winning rule applied
  verbatim to B. Best on A: `past_turn` top-10% = **-4.02%**. On held-out B the same rule
  gives **-1.83% (p=0.30, n=287)** vs -1.59% complement and -1.79% overall.
  **No subgroup has a materially larger pose benefit**; the -4.02% was selection noise.
  (Mild, non-significant trend: larger displacement / faster / turning agents benefit
  slightly more, ~-2.1 to -2.4% at moderate thresholds.) Reporting the A-number naively
  would have manufactured a fourth artifact of the kind this paper documents.
  Artifacts: `tools/scripts/subgroup_analysis.py`, `experiments/subgroup_analysis.json`.
- [x] **F4 SMPL 6D vs 3D skeleton keypoints — COMPLETE** (`POSE_REPRESENTATION=joints3d`).
  72-d root-relative SMPL-derived joints vs 144-d 6D rotations, same poses/model/protocol
  (`POSE_INPUT_DIM` makes the encoder input configurable; targets stay 6D so the decoder
  and inert aux losses are untouched). Verified: root joint exactly 0, max extent 0.81 m,
  masks identical to the 6d cell. This finally *tests* the paper's positioning claim
  ("parametric SMPL, unlike prior skeleton work"), which is currently only asserted.
  5 seeds, phase6. **RESULT: map_joints3d = 0.4807 +- 0.0049 (n=5), -0.97% vs no-pose
  (hier p=0.11); vs 6D rotations +0.61%, hier p=0.20 -> indistinguishable. The paper's
  positioning claim (parametric SMPL richer than skeleton keypoints) is NOT supported.**


## Part 6 — R2: intent-classification task (PRE-REGISTERED 2026-08-17, before any results)

**Motivation.** The skeleton/pose literature (fang2018pedestrian, rasouli2019pie,
Social-Transmotion-adjacent work) claims pose helps *crossing intent*, not 8s metric
forecasting. Test pose where the literature says the signal lives, on our data.

**Label (fixed before looking at any model output):** for each evaluated pedestrian,
using WORLD-frame GT: off-road at t0 (min distance from current position to any LANE
polyline point > 3.0 m) AND enters roadway within the future (min distance of any
valid GT future step to a lane polyline < 1.5 m) => positive "will-enter-roadway".
Population = pedestrians off-road at t0 with >= 10 valid future steps. Primary
horizon: full 8 s; secondary: 3 s. Parameters (3.0 / 1.5 m) chosen from lane
half-width reasoning, not tuned.

**Task & models.** Standalone binary classifier on the SAME dataset machinery:
trajectory-only encoder (GRU over past 11 agent-centric states) vs trajectory+pose
(adds the xattn+PE pose encoder used in the paper) vs trajectory+WRONG-pose (shuffled
donor control). Identical training protocol, 5 seeds each, train on training split,
evaluate AUROC on validation.

**Decision rule (pre-registered):** pose "helps intent" iff AUROC(pose) >
AUROC(no-pose) with hierarchical bootstrap p < 0.05 over 5 seeds AND
AUROC(pose) > AUROC(wrong-pose) at p < 0.05. Anything else = null carries over to the
intent task. No post-hoc threshold/horizon/population changes; secondary horizon is
reported alongside, not substituted.


## Part 7 — Temporal-denoising experiment (PRE-REGISTERED 2026-08-18, before any results)

**Question.** Is measurement noise (~160 mm root-aligned MPJPE, 230-260 mm extremities)
the binding factor behind the pose-content null, or is the content redundant/absent?

**Method.** Mask-aware temporal smoothing of the ENCODER-INPUT pose sequences only
(Gaussian kernel over valid frames, sigma = 1 frame, window 5; invalid frames untouched;
validity masks bit-identical; future-pose targets untouched). This attacks per-frame
jitter, which is what per-frame LiDAR-HMR error implies.

**Cells (5 seeds each, map condition, xattn+PE):**
  map_denoise          = correct poses, denoised
  map_denoise_shuffle  = WRONG (shuffled-donor) poses, denoised  [control]

**Decision rule:** "noise is the binding factor" iff map_denoise beats
map_denoise_shuffle at hierarchical p < 0.05 (a content signal emerges once jitter is
reduced). Secondary, reported not substituted: map_denoise vs map_xattn_pe (does
denoising help at all) and map_nopose -> map_denoise. Any other outcome => the
redundancy/absence explanation stands; no post-hoc smoothing-parameter search.

## Status log

- 2026-07-20: review completed; Part-1 fixes committed; this file created.
  2.1 launched; decisions taken (2.2 = everything to n=5; 2.3 = MR + Waymo mAP;
  2.4 = map_xattn_pe ×3). Mixing flaw found (Part 4); mixed runs renamed; pre-fix
  worktree created; 31-run serial queue launched (phase5 → 5b → 5c).

- 2026-07-21: review queue complete (34/34 OK). Final n=5 picture folded into paper
  (753ba7c), findings.md, research-state.yaml. Mechanism reversed (map benefit = body
  configuration, not heading). MR@2m in paper; mAP dropped per decision (b). PPTX next.
- 2026-08-12: professor follow-ups F1-F4 implemented (commit 74f46e0). F3 complete
  (negative result: no subgroup amplification survives held-out confirmation). F1/F4
  running (phase6), F2 chained (phase7). Two implementation confounds caught and fixed
  before launch: shuffle-induced pose-frame loss, and 8s intention points at a 2s horizon.
- 2026-08-12 (later): phase6 COMPLETE. F1 overturns the last positive claim (wrong poses
  recover ~75% of the benefit; correct-vs-wrong n.s.). F4: SMPL 6D == 3D keypoints (n.s.).
  Capacity control added (CONSTANT_POSE: identical pose for every pedestrian, zero
  information, masks unchanged) as phase8 to close the argument. phase7 (2s) rescheduled:
  a FOREIGN job (navsim_eval, 18.3 GB) took the GPU mid-run and OOM'd my runs; a
  free-GPU scheduler now waits for the card. A separate bug in the first 2s attempt was
  mine and is fixed (PREDICT_HORIZON_STEPS truncates the prediction horizon; changing
  NUM_FUTURE_TIMESTAMPS had desynced the 91-step data grid).
