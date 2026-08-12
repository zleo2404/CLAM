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

cd /scratch.hpc/leonardo.meloni/CLAM

WSI_DIR="/scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate"  
OUTPUT_DIR="/scratch.hpc/leonardo.meloni/CLAM/patching_results_l1"

python create_patches_fp.py \
    --source $WSI_DIR \
    --save_dir $OUTPUT_DIR \
    --patch_size 256 \
    --step_size 256 \
    --seg \
    --patch \
    --stitch \
    --preset tcga.csv \
    --patch_level 1

echo "--- Job completato ---"
echo "Data e ora: $(date)"