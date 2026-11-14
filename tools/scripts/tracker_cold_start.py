#!/usr/bin/env python
"""How much of the simulated tracker's velocity error is the one-second cold start?

WOMD gives one second of history, so the tracker of the onboard-track experiment
(WaymoPoseDataset._track_past_causally) is born at the first observed frame, at rest, and
never runs longer than 11 frames. A tracker on a vehicle has usually followed a pedestrian
for longer. This runs the dataset's own filter on a pedestrian walking at constant velocity
and reads its error after 11 frames, as in the paper, and after 3 s and 20 s, where it has
settled. The walker does not manoeuvre, so the numbers isolate detection noise and track age
and are below the errors measured on real pedestrians (tracker_state_error.py).

Writes experiments/tracker_cold_start.json.

Usage: python tools/scripts/tracker_cold_start.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset  # noqa: E402

OUT = os.path.join(REPO, 'experiments', 'tracker_cold_start.json')
SPEED, DT, N = 1.1, 0.1, 20000
AGES = {'1.1 s (11 frames, as in the paper)': 11, '3 s': 31, '20 s': 201}


def errors(noise_m: float, frames: int) -> dict:
    """Median error of the filter's last state on N constant-velocity walkers."""
    trajs = np.zeros((N, frames, 10))
    trajs[:, :, 0] = SPEED * DT * np.arange(frames)
    trajs[:, :, 7], trajs[:, :, 9] = SPEED, 1.0
    truth = trajs[:, -1].copy()
    stub = SimpleNamespace(position_noise_m=noise_m, dt=DT, num_past=frames)
    WaymoPoseDataset._track_past_causally(stub, trajs, list(range(N)), 'cold_start')
    last = trajs[:, -1]
    return dict(velocity_m_s=float(np.median(np.hypot(*(last[:, 7:9] - truth[:, 7:9]).T))),
                position_m=float(np.median(np.hypot(*(last[:, 0:2] - truth[:, 0:2]).T))))


def main() -> None:
    res = dict(_meta=dict(speed_m_s=SPEED, n_walkers=N, note=(
        'median error magnitude of the tracked state at the last frame, constant-velocity '
        'walker, the dataset\'s own filter (cold start at rest, sigma_a 1 m/s^2, sigma_v0 '
        '1.5 m/s). Track age is the number of frames the filter has run.')))
    for s in (0.1, 0.3):
        res[f'{s:g}'] = {name: errors(s, frames) for name, frames in AGES.items()}
        print(s, json.dumps(res[f'{s:g}']))
    json.dump(res, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
