#!/usr/bin/env python
"""Run tools/test.py on a listed subset of a split's scenes, beside a training job.

test.py scores every scene that has a pose directory, and passes all of a batch's
pedestrians through the model at once. A scene with 165 pedestrians then needs more GPU
memory than a training job on the same card leaves free. This wrapper changes three things
and nothing in the repository's dataset or model code:

  scenes   The whole split stays on disk and in the scene index while the dataset is built,
           so a stranger's pose is still drawn from all of it. Then only the listed scenes
           are kept.
  chunks   The model predicts each pedestrian from that pedestrian's own agent-centric
           scene, so a batch is passed through in chunks of CHUNK pedestrians and the
           predictions are concatenated. For the first batches that are small enough, both
           ways are run and their largest difference is logged (it is rounding error).
  memory   GPU memory is capped, so a shortage fails here and not in the training job.

Usage, from tools/:
  python scripts/eval_scene_subset.py <scene list> <GPU memory fraction> <test.py arguments>
Environment: CHUNK (default 16), CHUNK_CHECKS (default 20 batches).
"""
from __future__ import annotations

import os
import runpy
import sys

CHUNK = int(os.environ.get('CHUNK', '16'))
CHUNK_CHECKS = int(os.environ.get('CHUNK_CHECKS', '20'))


def main() -> None:
    scene_file, fraction = sys.argv[1], float(sys.argv[2])
    keep = set(open(scene_file).read().split())
    assert keep, f'{scene_file} lists no scenes'
    sys.argv = ['test.py'] + sys.argv[3:]
    sys.path.insert(0, os.getcwd())
    import _init_path  # noqa: F401  (puts the repository root on the path, as test.py does)
    import numpy as np
    import torch
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    from mtr.models.model import MotionTransformer

    build = WaymoPoseDataset.__init__
    log = {}

    def build_then_filter(self, *args, **kwargs):
        build(self, *args, **kwargs)        # the donor permutation is drawn in here
        n = len(self.infos)
        self.infos = [i for i in self.infos if i['scenario_id'] in keep]
        log['logger'] = self.logger
        self.logger.info(f'scene subset: {len(self.infos)} of {n} scenes kept '
                         f'({len(keep)} listed in {scene_file}), chunks of {CHUNK} pedestrians')
        assert self.infos, 'no listed scene is in this split'

    whole = MotionTransformer.forward
    state = dict(checked=0, worst=0.0)

    def part_of(inp: dict, n: int, lo: int, hi: int) -> dict:
        """The rows lo:hi of every per-pedestrian entry. Every tensor the model reads has
        the batch's pedestrians as its first axis."""
        out = {}
        for k, v in inp.items():
            per_ped = isinstance(v, (torch.Tensor, np.ndarray, list)) and len(v) == n
            out[k] = v[lo:hi] if per_ped else v
        return out

    def in_chunks(self, batch_dict):
        if self.training:
            return whole(self, batch_dict)
        inp = batch_dict['input_dict']
        n = len(inp['track_index_to_predict'])
        if n <= CHUNK:
            return whole(self, batch_dict)
        scores, trajs = [], []
        for lo in range(0, n, CHUNK):
            out = whole(self, dict(batch_dict, input_dict=part_of(inp, n, lo, lo + CHUNK)))
            scores.append(out['pred_scores'])
            trajs.append(out['pred_trajs'])
        scores, trajs = torch.cat(scores), torch.cat(trajs)
        assert len(scores) == n and len(trajs) == n
        if state['checked'] < CHUNK_CHECKS and n <= 2 * CHUNK:
            ref = whole(self, dict(batch_dict))
            d = max(float((ref['pred_trajs'] - trajs).abs().max()),
                    float((ref['pred_scores'] - scores).abs().max()))
            state['checked'] += 1
            state['worst'] = max(state['worst'], d)
            log['logger'].info(f'chunk check {state["checked"]}/{CHUNK_CHECKS}: {n} pedestrians, '
                               f'largest difference to one pass {d:.2e} (worst so far '
                               f'{state["worst"]:.2e})')
            assert d < 1e-2, f'chunked and whole predictions differ by {d}'
        batch_dict['pred_scores'], batch_dict['pred_trajs'] = scores, trajs
        return batch_dict

    WaymoPoseDataset.__init__ = build_then_filter
    MotionTransformer.forward = in_chunks
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    runpy.run_path('test.py', run_name='__main__')


if __name__ == '__main__':
    main()
