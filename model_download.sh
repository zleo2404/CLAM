#!/bin/bash
#SBATCH --job-name=HER2_FeatureExtraction
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


echo "--- Inizio job di Estrazione Patch (Step 2) ---"
echo "Data e ora: $(date)"

source /scratch.hpc/leonardo.meloni/clam_env/bin/activate

# Variabili di sicurezza
export HF_HOME="/scratch.hpc/leonardo.meloni/conda_home"
export TMPDIR="/scratch.hpc/leonardo.meloni/conda_home"
export XDG_CACHE_HOME="/scratch.hpc/leonardo.meloni/conda_home"

# Comando di download
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MahmoodLab/UNI2-h', max_workers=1)"

echo "--- Job completato ---"
echo "Data e ora: $(date)"