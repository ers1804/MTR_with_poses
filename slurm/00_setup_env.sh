#!/bin/bash
# Build the training env and compile MTR's CUDA extensions.
#
# Run INSIDE an interactive allocation (env builds are not allowed on login nodes):
#     salloc --gres=gpu:h100:1 --time=01:00:00
#     bash slurm/00_setup_env.sh
# or non-interactively:
#     srun --gres=gpu:h100:1 --time=01:00:00 bash slurm/00_setup_env.sh
#
# QUOTA: /hnvme allows ~82k files (soft) and ~33k are already used, while a torch conda env
# is 100k+ files. So the env is built in node-local $TMPDIR and stored in the workspace as a
# SINGLE conda-pack archive; training jobs unpack it back to their own $TMPDIR.
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
BUILD=${TMPDIR:-/tmp}/mtr_env_build

module add gpu-env/2025
module add python/3.12-conda
module add cuda/12.6.2

export http_proxy=http://proxy.nhr.fau.de:80  https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$BUILD/pkgs"          # node-local: no quota, no $HOME writes
export PIP_CACHE_DIR="$BUILD/pipcache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$WS/env"

echo "=== creating env in $BUILD/env (node-local) ==="
conda create -y -p "$BUILD/env" python=3.10
source activate "$BUILD/env"

pip install --upgrade pip
# Versions pinned to the workstation setup so cluster numbers stay comparable to the
# 579-scene and 5k-scene rungs (torch 2.3.1+cu121, numpy <2).
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 easydict pyyaml tensorboardX tqdm scikit-image scikit-learn \
            scipy pandas opencv-python-headless protobuf wandb ninja conda-pack

echo "=== compiling CUDA extensions (knn_cuda, attention_cuda) ==="
cd "$WS/MTR_with_poses"
# NOT `setup.py develop`: modern setuptools routes it through an isolated PEP 517 build that
# has no torch installed, so setup.py's `from torch.utils.cpp_extension import CUDAExtension`
# dies with ModuleNotFoundError. Build against the env's own torch instead.
pip install -e . --no-build-isolation
python - <<'PY'
import torch
from mtr.ops.knn import knn_cuda            # submodules, not top-level imports
from mtr.ops.attention import attention_cuda
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("CUDA extensions import OK")
PY
rm -rf "$WS/MTR_with_poses/build"       # keep the .so files, drop object-file clutter

echo "=== packing env to a single workspace file ==="
conda pack -p "$BUILD/env" -o "$WS/env/mtr_smpl.tar.gz" --force
ls -lh "$WS/env/mtr_smpl.tar.gz"
echo "ENV PACKED. Training jobs unpack this to their own \$TMPDIR."
