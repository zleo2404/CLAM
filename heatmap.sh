#!/bin/bash
#SBATCH --job-name=clam_heatmap
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/leonardo.meloni/CLAM
#SBATCH --output=log_output_%j.txt
#SBATCH --error=log_error_%j.txt

echo "--- Inizio Generazione Heatmap CLAM (HER2) ---"
echo "Data e ora: $(date)"

# Attiva l'ambiente virtuale
source /scratch.hpc/leonardo.meloni/clam_env/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TIMM_OFFLINE=1

export UNI_CKPT_PATH="/scratch.hpc/leonardo.meloni/conda_home/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config uni_clamb.yaml

echo "--- Generazione completata ---"
echo "Data e ora: $(date)"