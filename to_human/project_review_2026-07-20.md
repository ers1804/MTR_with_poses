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
- [ ] **2.3 External metrics anchor** — no miss rate / Waymo mAP / published-baseline comparison.
  (scope TBD: offline MR@2m from stored per-agent minFDE is cheap; full Waymo mAP heavy;
  Social-Transmotion comparison = separate codebase, likely out of scope)
- [ ] **2.4 World-frame pose confound** — trajectories are rotated agent-centric but pose features
  are not. Plan: dataset flag to rotate root orientation into the agent frame; 3 seeds on a chosen
  cell; compare. (cell choice TBD)
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

## Status log

- 2026-07-20: review completed; Part-1 fixes committed; this file created.
  2.1 launched. 2.2 (norootorient/map_wta01 404/505) queued behind 2.1.
