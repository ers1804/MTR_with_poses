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
# smplpytorch is a TOP-LEVEL import in mtr/models/motion_decoder/mtr_decoder.py, so training
# cannot even import without it — despite the SMPL layer only feeding the (inert) MPJPE loss.
# chumpy is its dependency and needs build isolation off: its setup.py imports numpy.
pip install chumpy==0.70 --no-build-isolation
pip install smplpytorch==0.0.8 --no-deps
# chumpy 0.70 does `from numpy import bool, int, float, ...`, aliases numpy removed in 1.24,
# so importing it raises ImportError. The workstation env has this same one-line patch applied
# by hand; replicate it here or the decoder cannot import smplpytorch.
python - <<'PATCH'
import pathlib, re, site
for sp in site.getsitepackages():
    f = pathlib.Path(sp) / "chumpy" / "__init__.py"
    if f.exists():
        txt = f.read_text()
        fixed = re.sub(r"^from numpy import .*\bbool\b.*$", "from numpy import nan, inf",
                       txt, count=1, flags=re.M)
        if fixed != txt:
            f.write_text(fixed); print("patched", f)
        else:
            print("no patch needed", f)
PATCH
python -c "import chumpy; from smplpytorch.pytorch.smpl_layer import SMPL_Layer; print('smplpytorch import OK')"
# tensorflow / waymo-open-dataset are imported only by data_preprocess.py and waymo_eval.py,
# neither of which is on the training path (the workstation env does not have them either).

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
# conda-pack refuses envs containing editable installs. Ours (motiontransformer) points at
# $WS/MTR_with_poses, a persistent path that exists on every compute node, so the .pth
# reference still resolves after the env is unpacked into a job-local $TMPDIR.
conda pack -p "$BUILD/env" -o "$WS/env/mtr_smpl.tar.gz" --force --ignore-editable-packages
ls -lh "$WS/env/mtr_smpl.tar.gz"
echo "ENV PACKED. Training jobs unpack this to their own \$TMPDIR."
