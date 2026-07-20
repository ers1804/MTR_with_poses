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

- [ ] **2.1 Root-orientation ablation in the MAP condition** ← *highest value; IN PROGRESS 2026-07-20*
  The mechanistic headline ("even the surviving −1.6% is mostly heading re-entry") is extrapolated
  from the no-map ablation to the map×xattn+PE cell where it was never tested.
  Plan: `mtr+pose_data_cross_attn_pe_with_map_norootorient.yaml`, 3 seeds (101/202/303),
  tag `MS_map_norootorient_s<seed>`. Decision rule: if map_norootorient ≈ map_nopose, the heading
  story extends to the map condition; if map_norootorient ≈ map_xattn_pe, the map-condition gain is
  NOT heading-driven and §4.4/§4.5/Limitations/Conclusion must be softened.
- [ ] **2.2 n=3 → n=5 for cells supporting claims** (self-consistency with the paper's own thesis)
  - [ ] `norootorient` seeds 404/505 (σ=0.0031 suspiciously tight — the exact artifact signature)
  - [ ] `map_wta01` seeds 404/505 (supports "GRU+map gives nothing")
  - [ ] (scope TBD) `pose30fps`, masked audit cells to n=5
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
4. After completion: regenerate analysis; **the xattn_pe collapse claim must be
   re-evaluated on the clean pre-fix 5-seed cell** — if pre-fix 404/505 come out tight,
   the "second artifact" story changes again and the paper must be updated accordingly.
5. Note for mixed-comparator pairs: map_nopose/ft_nopose 404/505 (phase4e) are post-fix,
   but those cells are no-pose; the fix was verified to leave the no-pose baseline
   unchanged (p=0.95), so treating them as one cell is defensible — disclosed here.

## Status log

- 2026-07-20: review completed; Part-1 fixes committed; this file created.
  2.1 launched; decisions taken (2.2 = everything to n=5; 2.3 = MR + Waymo mAP;
  2.4 = map_xattn_pe ×3). Mixing flaw found (Part 4); mixed runs renamed; pre-fix
  worktree created; 31-run serial queue launched (phase5 → 5b → 5c).
