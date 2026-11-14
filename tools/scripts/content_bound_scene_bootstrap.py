#!/usr/bin/env python
"""Does the bound on the 25k content test survive resampling the validation scenes?

The paper bounds the content effect with a paired interval over five seeds,
[-1.59%, +1.70%], on one fixed validation set. That interval carries the seed-to-seed
variation and none from the choice of scenes. This adds it: the content effect is recomputed
on validation sets resampled by scene, with the seeds fixed, and with seeds and scenes
resampled together.

Reads the per-agent metrics of the 25k pose and shuffled runs at each run's best epoch
(pointcount_stratified_content.py). Writes experiments/content_bound_scene_bootstrap.json.

Usage: python tools/scripts/content_bound_scene_bootstrap.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from pointcount_stratified_content import CELLS, REPO, SEEDS, load_run, paired

OUT = os.path.join(REPO, 'experiments', 'content_bound_scene_bootstrap.json')
N_BOOT = 10000


def main() -> None:
    runs = {a: {s: load_run(*CELLS[a], s) for s in SEEDS} for a in ('pose', 'shuf')}
    assert all(v is not None for r in runs.values() for v in r.values()), 'a run is missing'
    agents = sorted(set.intersection(*[set(v) for r in runs.values() for v in r.values()]))
    scenes = sorted({a[0] for a in agents})
    of = {s: i for i, s in enumerate(scenes)}
    idx = np.array([of[a[0]] for a in agents])
    # per-scene sums of minADE, one row per seed, so a resampled set is a weighted sum
    S = {a: np.stack([np.bincount(idx, weights=np.array([runs[a][s][x] for x in agents]),
                                  minlength=len(scenes)) for s in SEEDS]) for a in runs}

    def effect(w: np.ndarray, seeds: np.ndarray) -> float:
        """Mean over seeds of the relative difference of arm means, scenes weighted by w."""
        p, q = S['pose'][seeds] @ w, S['shuf'][seeds] @ w
        return float(((p - q) / q * 100).mean())

    ones, every = np.ones(len(scenes)), np.arange(len(SEEDS))
    seed_only = paired(runs, 'pose', 'shuf', agents, np.ones(len(agents), bool))
    assert abs(effect(ones, every) - seed_only['rel']) < 1e-9
    rng = np.random.default_rng(0)
    sc = np.array([effect(np.bincount(rng.integers(len(scenes), size=len(scenes)),
                                      minlength=len(scenes)).astype(float), every)
                   for _ in range(N_BOOT)])
    both = np.array([effect(np.bincount(rng.integers(len(scenes), size=len(scenes)),
                                        minlength=len(scenes)).astype(float),
                            rng.integers(len(SEEDS), size=len(SEEDS)))
                     for _ in range(N_BOOT)])
    res = dict(
        _meta=dict(n_agents=len(agents), n_scenes=len(scenes), seeds=list(SEEDS), n_boot=N_BOOT,
                   note='content effect in percent, own pose against shuffled, 25k scenes, each '
                        'run at its best epoch. Negative favours own pose. seed_interval is the '
                        'paired t interval of the paper. The bootstraps are percentile intervals.'),
        content=seed_only['rel'], seed_interval=[seed_only['lo'], seed_only['hi']],
        scenes_resampled=[float(x) for x in np.percentile(sc, [2.5, 97.5])],
        seeds_and_scenes_resampled=[float(x) for x in np.percentile(both, [2.5, 97.5])])
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
