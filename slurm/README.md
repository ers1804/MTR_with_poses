# Helma (NHR@FAU) — pose scale-ladder sweep

Slurm cluster, workspaces on NVMe (`/hnvme/workspace/<user>-<id>`), h100/h200 partitions.

## One-time setup

1. **Workspace** (Erik creates, or):
   `ws_allocate mtr_smpl 60`  → `/hnvme/workspace/v103fe12-mtr_smpl`
   Export `WS=<that path>` if the id differs; every script takes `WS` as an env var.

2. **Code**: `git clone git@github.com:ers1804/MTR_with_poses.git $WS/MTR_with_poses && cd $WS/MTR_with_poses && git checkout mtr_smpl`

3. **Environment + CUDA extensions** (needs a GPU node — `knn_cuda`/`attention_cuda` are
   compiled CUDA extensions, this is the step most likely to need fixing):
   `sbatch slurm/00_setup_env.sh`

4. **Data**: shipped by `/home/erik/big1/helma_stage/pack_and_ship.sh` from the workstation
   into `$WS/data/` as `poses_25k.tar` (~1.7 GB) + `scenarios_25k.tar` (~37 GB) +
   `cluster_64_center_dict*.pkl`. Jobs untar to node-local `$TMPDIR`, never read them
   from the shared filesystem during training.

## Running the sweep

`sbatch slurm/10_scale_sweep.sh`  → 27-task array (3 cells × 3 LRs × 3 seeds), 12 concurrent.

Cells: `no_pose_with_map`, `cross_attn_pe_with_map`, `cross_attn_pe_with_map_shufflepose`.
LRs: 1e-4, 5e-5, 2.5e-5 — **swept per arm on purpose**: at 4.9× data a single inherited LR
produced a +37% fake "pose hurts" effect that vanished (−0.9%) once each arm was trained
where it converges (research-log 2026-08-21/22). Compare per-arm best-of-sweep, never a
single shared LR.

## Pulling results back

Runs write `output/waymo/<cfg>/HELMA_<rung>_<cell>_lr<lr>_s<seed>/`; checkpoints are deleted
at job end, logs+metrics kept. Sync back with:
`rsync -av helma:$WS/MTR_with_poses/output/waymo/ <local>/output/waymo/`
then add cells to `tools/scripts/analyze_multiseed.py` and run the hierarchical bootstrap.

## Cluster-specific constraints (learned the hard way)

- **`$TMPDIR` is wiped after every job.** Nothing durable may live there. Each job re-stages
  the env archive and the data tars into it; outputs go to `$WS/MTR_with_poses/output/`.
- **File quota on /hnvme: ~82k soft / 102k hard, ~33k already used.** A torch conda env is
  100k+ files on its own, so the env is stored as ONE `conda-pack` archive and the datasets
  stay as two tars. Never extract either into the workspace.
- **Env builds only inside an allocation** (`salloc --gres=gpu:h100:1 --time=01:00:00`), not
  on login nodes.
- The compiled CUDA extensions (`.so`) live in the workspace repo and are ABI-bound to the
  packed env's torch 2.3.1 — rebuild both together if either changes.

## Gotchas carried over from the workstation

- `PYTHONNOUSERSITE=1` always (a `~/.local` torch shadows conda envs).
- `TRAIN_MAX_CENTERS=24`: clean-export scenes reach 153 pedestrians; uncapped scenes OOM.
- Validation set must stay the fixed 5,171-scene split for comparability with all prior runs.
- Batch size 2 in the `_scale5k_lr5` configs was chosen for a 24 GB card; on 80 GB H100 it
  can go up, but **changing it changes the effective recipe** — if you raise it, re-tune LR
  and treat the rung as a new experiment rather than comparing to the workstation numbers.

## Talking to Helma from automation: use slurm/helma_ssh.sh

    slurm/helma_ssh.sh 'squeue -u $USER'      # any command, over a reused connection
    slurm/helma_ssh.sh --status               # queue snapshot
    slurm/helma_ssh.sh 'cat $WS/logs/status.txt'   # whole-array progress in ONE round trip

It multiplexes (one master connection reused for 8 h, so repeat checks cost no handshakes on
Helma *or* on the csnhr jump host), rate-limits new master connections to one per 2 min with a
lock so parallel watchers cannot stampede, and backs off progressively after refusals.

Sweep jobs also self-report into `$WS/logs/status.txt` (START / DONE+best / EXIT rc), so the
whole 27-task array can be checked with a single cheap read instead of per-task Slurm queries.

## Do not poll the login node from a loop

Watching jobs by opening a fresh `ssh helma` every ~60 s (several watchers in parallel) got the
login node to refuse connections outright (`kex_exchange_identification: Connection closed by
remote host`). Running jobs are unaffected — they live on compute nodes — but you lose visibility
until the throttle clears.

Use one multiplexed connection instead, and poll at minutes, not seconds:

    ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h -o ControlPersist=60m helma true
    # subsequent commands reuse the socket:
    ssh -o ControlPath=~/.ssh/cm-%r@%h helma 'squeue -u $USER'

Better still for long sweeps: have the job itself write a status line into `$WS/logs/` and check
that file every 10+ minutes, rather than querying Slurm repeatedly.

## Watchdog / restarting failed tasks

    bash slurm/30_check_and_resubmit.sh            # report only
    bash slurm/30_check_and_resubmit.sh --apply    # resubmit dead tasks

A task is complete when its output dir holds a `DONE` marker (written only after training AND
eval succeeded). Anything missing that is *not* currently queued or running counts as dead —
node failure, timeout, preemption, OOM — and is resubmitted by index, so nothing already
finished is recomputed. Safe to run repeatedly; also handles both LR grids (main + extension).

Run it from cron every couple of hours during a sweep. Watch for the same index dying
repeatedly: after two resubmits, stop and diagnose rather than spending more allocation.

## Frame / architecture waves (review follow-up, 2026-08-26)

    sbatch --array=0-8 slurm/40_frame_arch_sweep.sh wave1   # coordinate-frame fix + control
    sbatch --array=0-8 slurm/40_frame_arch_sweep.sh wave2   # derived descriptors, early fusion + control
    bash slurm/41_check_frame_arch.sh [--apply]             # watchdog for both waves

The task grid lives ONLY in `40_frame_arch_sweep.sh` as explicit `cfg|lr|seed` strings; the
watchdog parses it back out, so the two cannot drift apart. Status goes to `$WS/logs/status_fa.txt`.

wave1 sets `AGENT_CENTRIC_POSE_ROT`, which rotates the pose root orientation into the center
agent's frame — the frame the trajectory features already use. wave2 replaces the raw 144-d pose
with 8 frame-invariant descriptors concatenated onto the agent's per-timestep features and turns
the pose branch OFF (`USE_POSE_ENCODER: False`, `NUM_INPUT_ATTR_AGENT: 37`), so it adds input
channels rather than an encoder — relevant because the branch itself measured +1.7% cost at 25k.

Read results against the 25k rung (no-pose 0.3711 / pose 0.3774 / shuffled 0.3770) and judge on
CORRECT vs ITS OWN CONTROL.
