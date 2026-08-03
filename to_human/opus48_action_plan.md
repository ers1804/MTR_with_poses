# Action Plan for Opus 4.8 — MTR_with_poses (branch `mtr_smpl`)

Date: 2026-07-08. Produced from a three-way review (model/loss code, branch diffs + statistics, paper-vs-data consistency).

## Executive summary

The 2026-06-12 audit caught the inert geodesic loss, but **the same bug class is still live**: there is *no future-pose validity mask anywhere in the pipeline*, so every pose-supervision loss on 10fps data trains against zero targets. `mpjpe` is a **T-pose regularizer with full gradient** (zero 6D → Gram-Schmidt → zero rotation → T-pose joints), and the WTA-L1 ("gmm_pose") actively pulls predictions toward the zero 6D vector through the shared decoder trunk. The paper's "auxiliary-loss audit" axis therefore still doesn't measure pose supervision — it measures zero-target regularizers. Additionally, the **past** pose mask is trajectory validity, not pose validity, so the encoder (the mechanism the paper credits for the trajectory gain) ingests zero-poses flagged as valid.

On the paper side, all table numbers match `experiments/multiseed_analysis.json`, but there are two outright errors (run-011 Δ sign; the false "every seed improves" claim), an 8.1s vs 9.0s horizon contradiction, a ~1.3-page overage vs the ICLR 9-page limit, and a statistical protocol whose CIs marginalize seeds while the text claims seed-level significance.

The single biggest scientific gap: **the stable winning encoder (xattn+PE, −3.5%) was never run in the realistic condition (map / map+pretrain)** — those 2×2s use the GRU, the encoder the paper itself shows is heavy-tailed and on-average worse than baseline. `tools/cfgs/waymo/mtr+full_ped_finetune_geo.yaml:57` confirms GRU. Six runs (~10–15 min each) close this.

Also a reproducibility breaker: the blanket `*.yaml` gitignore means **8 configs the paper depends on are untracked** — a fresh clone cannot run a single headline cell.

---

## Phase 0 — Hygiene & reproducibility (no risk, do first)

1. `git add -f` (or `!tools/cfgs/**/*.yaml` gitignore exception) for the 8 ignored configs: `mtr+pose_data_no_pose.yaml`, `geo_pure`, `cross_attn`, `no_pose_with_map`, `geo_only_with_map`, `full_ped_finetune_geo`, `full_ped_finetune_no_pose`, `full_ped_pretrain`.
2. `git rm --cached` LaTeX build artifacts (`paper/iclr2026/main.{aux,bbl,blg,log,out}`), delete stray root `main.log` (committed in 13b2201) and `texput.log`; add `*.aux/*.bbl/*.blg/*.out/texput.log` to `.gitignore`; gitignore `.claude/`.
3. Commit `tools/scripts/{analyze_multiseed.py,run_multiseed_matrix.sh,run_multiseed_phase2.sh}` (chmod +x the .sh) and `experiments/multiseed_analysis.json`.
4. Change `analyze_multiseed.py` default `--out` from `/tmp/multiseed_analysis.json` to `experiments/multiseed_analysis.json`; fix findings.md/research-state.yaml pointers.

## Phase 1 — Correctness fixes (code)

**P1.1 Future-pose validity mask (root cause; highest impact).**
Build `center_gt_poses_mask` = (pose row nonzero) ∧ (traj step valid) in `waymo_pose_dataset.py` (`_load_pedestrian` / `create_scene_level_data`, around lines 549–554) and thread it through:
- WTA mode selection + WTA-L1 normalization (`mtr/utils/loss_utils.py:100–112`) — divide by valid count;
- `get_mpjpe_loss` (`mtr_decoder.py:476–485`, applied :607–611) — currently **no mask at all**;
- geodesic loss (`mtr_decoder.py:615–618`) — currently unmasked;
- `cls_pose` gating (`mtr_decoder.py:622`).
When an agent has zero valid future-pose steps, its pose losses must be exactly 0 (and excluded from denominators).

**P1.2 Past-pose mask.** `waymo_pose_dataset.py:543–547`: `obj_poses_mask` must be pose-row-nonzero ∧ traj-valid, not traj-valid alone. Consider `pack_padded_sequence`/last-valid gather so the GRU final hidden state isn't taken after trailing zero frames.

**P1.3 Eval robustness.** `waymo_pose_dataset.py:968–970` (`per_agent_metrics`): add the `len(last_valid_idx) > 0` guard that `evaluation()` (line 918) has — one bad agent currently kills the whole `metrics_epoch_N.pkl` via the caller's try/except (`tools/eval_utils/eval_utils.py:82–87`). Also: restrict the `__getitem__` random-resample-on-error (`waymo_pose_dataset.py:213–219`) to training mode; re-raise in eval.

**P1.4 Best-model tracking.** `waymo_pose_dataset.py:929` returns `'mAP': 0.0`, and `train_utils.py:188–199` prefers mAP → `best_model.pth` is frozen at the first evaluated epoch. Remove the placeholder or make tracking prefer `minADE_TYPE_PEDESTRIAN` (lower-is-better).

**P1.5 Silent-config hazards.** `mtr_decoder.py:547–550`: change `LOSS_WEIGHTS.get(pose_key, 1.0)` defaults to `0.0` (a typo currently enables a pose loss at full weight). `tools/train.py:164`: `wandb.init(id=args.extra_tag)` collides across configs and resurrects deleted runs — use `id=f"{cfg.TAG}-{args.extra_tag}"` or drop `id`.

**P1.6 Dataset guards.** `waymo_pose_dataset.py:406–408, 440–441`: grid mapping via `argmin` needs `|time_grid[gi] − ts| < dt/2` tolerance (current bounds check is dead code — same bug class as the April µs incident). 30fps branch (:392–398, 417–430): assert `start_idx >= 0` and uniform spacing (negative start silently writes to the array tail via Python negative indexing).

**P1.7 cls-loss gating.** `mtr_decoder.py:634`: gate `loss_cls` (and `loss_cls_pose`) on agents with ≥1 valid future step; unjoined pedestrians (~20%) currently get goals at the origin.

## Phase 2 — Statistics upgrade

1. **Hierarchical bootstrap** in `analyze_multiseed.py`: resample seeds, then pedestrians (or report a seed-level test alongside). Current CIs marginalize seeds; at seed level, "geo_pure significantly better (−0.9%)" is p≈0.7 by t-test. The paper's Limitations (main.tex:809) claims the CIs capture what they don't.
2. Floor bootstrap p at `1/n_boot` (JSON currently stores literal `0.0`); cap `2*min(...)` at 1.0.
3. Rename/fix `epoch30_minADE` → last-evaluated-epoch, and add an epoch-count sanity check so a died-early run can't pollute a cell mean.
4. Note multiplicity: 18 pairs uncorrected; p=0.046 (map pose effect) must stay labeled "marginal".

## Phase 3 — Paper fixes (`paper/iclr2026/main.tex`)

Numeric/factual (do not touch verified numbers — all table values match the JSON):
- **Line ~1030**: run 011 (geo w=0.05, 0.6786) vs baseline 0.6745 is **+0.6% worse**, not "−1.5%".
- **Lines ~671–672**: "every seed improves" is false — seed 202: ft_nopose .3753 → ft_wta01 .3760 (worse). Reword to 2-of-3 seeds.
- **Lines ~369/393 vs 890/909**: "8.1 s" is wrong; 91 steps @10Hz span 9.0 s, future horizon = 8.0 s.
- Line ~410 vs 631: unify "≈487k" vs "≈486k" (486,995).
- Line ~120: parenthetical misattaches 0.6604 to the GRU; it's the baseline mean.
- Lines ~456, 621–623: "order of magnitude" → 5.5×; "σ up to 0.092" matches no table a reader can find (Table 7 has 0.1354).
- Line ~403: "three random seeds" → "three to five"; reproducibility statement (~856) should mention seeds 404/505.
- Hedge the pedestrian-bootstrap "significance" language per Phase 2, or update after hierarchical bootstrap.
- **Page limit**: cut ~1.3+ pages (target 9 main pages): §5/Table 7 overlap with Tables 1–2 is the cut candidate.
- Cosmetic: Table 6 run ordering (019 before 018), overfull hbox Table 2, `[h]`→`[htbp]` on appendix floats.

## Phase 4 — New experiments (ranked by value/cost)

1. **xattn+PE × {map, map+pretrain} × 3 seeds (6 runs, ~1.5 h GPU total).** Create `mtr+pose_data_cross_attn_pe_with_map.yaml` and `mtr+full_ped_finetune_xattn_pe.yaml` (xattn+PE + `POSE_FUSER_RESIDUAL` analog for the xattn path — verify zero-init applies there too). Closes the paper's biggest hole: the stable encoder never meets the realistic condition; the −1.9% headline uses the encoder the audit discredits.
2. **Root-orientation ablation (3–6 runs).** Zero/noise the root-orientation channel of the 144-dim pose input (keep body joints). Controls the heading-snapping circularity a reviewer will attack: is the pose benefit just trajectory heading re-entering through a side door?
3. **Seeds 404/505 for xattn_pe** — the "stable winner, σ=0.0026" claim rests on n=3.
4. **After P1.1/P1.2 land: re-run the loss-audit cells (wta01, mpjpe, full, geo_pure) with real masks**, 3 seeds each — only then does the auxiliary-loss axis measure supervision rather than zero-target regularization. Compare against old cells; the paper's audit section gets strictly stronger either way.
5. Optional/stretch: agent-centric rotation of root orientation (`waymo_pose_dataset.py:695–719` rotates trajs but not poses — pose features are rotation-variant); eval-time pose metrics (predictions currently discarded in `generate_final_prediction`, `mtr_decoder.py:714–730`); standard Waymo metrics (mAP/miss rate) for the pretrained cells to anchor externally.

## Verification protocol (apply to every phase-1 change)

- Unit-test the masks: construct a batch with a known pattern of valid/invalid pose steps; assert loss gradients are exactly zero w.r.t. predictions at invalid steps and nonzero at valid ones (this is the test that would have caught both the geo and mpjpe bugs).
- One smoke run per changed config (2 epochs) before any matrix relaunch; confirm `metrics_epoch_N.pkl` written and loss magnitudes sane.
- Re-run `analyze_multiseed.py` after statistics changes and diff `experiments/multiseed_analysis.json`; any changed headline number must be propagated to findings.md, research-state.yaml, and main.tex together (this repo's history shows numbers drift across the three).
- Recompile paper; check page count and `main.log` for new warnings.

---

## Implementation prompt for Opus 4.8

Copy-paste everything below the line into a fresh Opus 4.8 session in this repo.

---

You are working in `/home/erik/ssd2/gitprojects/MTR_with_poses` on branch `mtr_smpl`. This is a research codebase (MTR trajectory prediction + SMPL pose conditioning, ICLR 2026 draft in `paper/iclr2026/main.tex`). Read `to_human/opus48_action_plan.md` in full first — it contains a reviewed action plan with file:line references; treat it as the specification. Then read `findings.md` (especially the "CRITICAL CORRECTION" section) and the `multiseed_matrix` section of `research-state.yaml` for context.

Execute the plan in this order, committing after each phase with conventional-commit messages:

**Phase 0 (hygiene):** Fix the gitignore so the 8 experiment configs listed in the plan are tracked; untrack LaTeX build artifacts; delete stray logs; commit the multiseed scripts and `experiments/multiseed_analysis.json`; change the analyze script's default output path to the repo. Verify with `git status` that a fresh clone would contain every config named in `tools/scripts/run_multiseed_matrix.sh`.

**Phase 1 (correctness):** Implement fixes P1.1–P1.7 exactly as specified in the plan. The core change is a future-pose validity mask (`center_gt_poses_mask` = pose-row-nonzero ∧ traj-valid) created in `mtr/datasets/waymo/waymo_pose_dataset.py` and consumed by all four pose losses in `mtr/models/motion_decoder/mtr_decoder.py` and `mtr/utils/loss_utils.py`, plus fixing the past-pose mask `obj_poses_mask` to require pose-row non-zeroness. Before writing fixes, write the failing test: a pytest in `test/` that builds a synthetic batch with known valid/invalid pose steps and asserts (a) loss gradient w.r.t. predictions at invalid steps is exactly zero, (b) gradient at valid steps is nonzero, (c) an agent with zero valid future-pose steps contributes exactly zero pose loss. This test must fail against current code (it will — mpjpe is unmasked) and pass after. Do not change any loss semantics beyond masking/normalization; the multi-seed results must remain comparable where the plan says they remain valid.

**Phase 2 (statistics):** Add a hierarchical bootstrap (resample seeds, then pedestrians) to `tools/scripts/analyze_multiseed.py` alongside the existing paired bootstrap; floor p-values at 1/n_boot; fix the `epoch30_minADE` label and add an epoch-count sanity check. Re-run the script against the existing `output/waymo/` logs and write the result to `experiments/multiseed_analysis.json`. Report in your summary which conclusions survive the hierarchical CIs and which weaken (expect: xattn_pe −3.5% and ft pose −1.9% survive; geo_pure −0.9% and map −0.45% likely lose significance).

**Phase 3 (paper):** Apply the listed factual fixes to `paper/iclr2026/main.tex` (run-011 sign error, false "every seed improves" claim, 8.1s→8.0/9.0s horizon, 487k/486k, misattached parenthetical, "order of magnitude"→5.5×, seed-count wording, reproducibility statement seeds 404/505). Update significance language to match the Phase-2 hierarchical results. Then reduce main content to ≤9 pages, cutting the §5/Table 7 redundancy first. Recompile with the repo's usual pdflatex/bibtex cycle; verify zero undefined references and report the final page count. Update `findings.md` and `research-state.yaml` in the same commit as any number that changes — the three files must never disagree.

**Phase 4 (experiments) — only if GPU access is confirmed; otherwise stop after Phase 3 and print the exact launch commands instead.** Create configs for xattn+PE with map (`mtr+pose_data_cross_attn_pe_with_map.yaml`) and xattn+PE finetune from the H7 checkpoint (`mtr+full_ped_finetune_xattn_pe.yaml`, verifying the residual zero-init pose_fuser path also covers the cross-attention encoder — check `mtr/models/context_encoder/mtr_encoder.py` and add it if it only covers the GRU path). Launch 3 seeds (101/202/303) each via the same pattern as `tools/scripts/run_multiseed_matrix.sh`, tags `MS_map_xattn_pe_s<seed>` and `MS_ft_xattn_pe_s<seed>`. Also create the root-orientation-zeroed ablation config and launch 3 seeds. After runs finish, extend `analyze_multiseed.py`'s cell list, regenerate the analysis JSON, and report the new pairs: map_nopose→map_xattn_pe and ft_nopose→ft_xattn_pe.

Constraints: never modify numbers in the paper without a source in `experiments/multiseed_analysis.json` or a run log; do not delete or rename existing output directories; do not push; run the pose-mask pytest and a 2-epoch smoke run of `mtr+pose_data.yaml` before committing Phase 1 (use `tools/train.py --cfg_file tools/cfgs/waymo/mtr+pose_data.yaml --extra_tag SMOKE_maskfix --epochs 2 --batch_size 10 --random_seed 101` and delete the SMOKE output dir afterwards). If data paths under `/home/erik/NAS/` are unavailable, say so and skip smoke runs rather than faking them. End with a summary listing: files changed per phase, test results, page count, and which paper claims changed.
