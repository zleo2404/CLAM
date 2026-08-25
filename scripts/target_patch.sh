#!/bin/bash
#SBATCH --job-name=HER2_FeatureExtraction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/leonardo.meloni/CLAM
#SBATCH --output=log_output_%j.txt
#SBATCH --error=log_error_%j.txt

echo "--- Inizio job di Estrazione Feature con UNI (Modalità Offline) ---"
echo "Data e ora: $(date)"

source /scratch.hpc/leonardo.meloni/clam_env/bin/activate

cd /scratch.hpc/leonardo.meloni/CLAM/scripts

python extract_target.py

echo "--- Job completato ---"
echo "Data e ora: $(date)"