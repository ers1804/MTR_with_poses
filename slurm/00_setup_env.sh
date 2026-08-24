#!/bin/bash -l
#SBATCH --job-name=mtrpose_env
#SBATCH --output=%x_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
# One-time: create the conda env and COMPILE MTR's CUDA extensions (knn_cuda,
# attention_cuda). Must run on a GPU node so the build sees a real device arch.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_pose}
ENV_NAME=${ENV_NAME:-mtr_smpl}

module add python/3.12-conda
module add cuda/12.6.2
export PYTHONNOUSERSITE=1
# HOME sits at ~96G of 100G and 483k of 500k inodes. Keep every cache off it, or env
# creation dies partway through with quota errors.
export CONDA_PKGS_DIRS="$WS/.conda_pkgs"
export PIP_CACHE_DIR="$WS/.pip_cache"
export TORCH_EXTENSIONS_DIR="$WS/.torch_extensions"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
export http_proxy=http://proxy.nhr.fau.de:80  https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy

# env lives in the workspace (home quota is small, and NVMe is fast)
conda create -y -p "$WS/envs/$ENV_NAME" python=3.10
conda activate "$WS/envs/$ENV_NAME"

pip install --upgrade pip
# torch build must match the loaded CUDA toolkit family for the extension build
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.4 easydict pyyaml tensorboardX tqdm scikit-image scikit-learn \
            scipy pandas opencv-python-headless protobuf wandb ninja

cd "$WS/MTR_with_poses"
python setup.py develop            # builds knn_cuda + attention_cuda

python - <<'PY'
import torch, knn_cuda, attention_cuda
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
print("CUDA extensions import OK")
PY
echo "ENV READY: $WS/envs/$ENV_NAME"
