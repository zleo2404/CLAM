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
#SBATCH --chdir=/scratch.hpc/leonardo.meloni/HER2
#SBATCH --output=slurm/log_output_%j.txt
#SBATCH --error=slurm/log_error_%j.txt

echo "--- Inizio job di Segmentazione CLAM (Step 1) ---"
echo "Data e ora: $(date)"

# 1. Attiva l'ambiente virtuale dedicato a CLAM
source /scratch.hpc/leonardo.meloni/clam_env/bin/activate

# 2. Spostati nella cartella di CLAM
cd /scratch.hpc/leonardo.meloni/CLAM

# 3. Definisci i percorsi (Assicurati che questi percorsi esistano!)
# WSI_DIR: cartella dove si trovano i tuoi file .svs / .ndpi / .tif
WSI_DIR="/scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate"  

# OUTPUT_DIR: dove CLAM salvera le maschere e il file CSV
OUTPUT_DIR="/scratch.hpc/leonardo.meloni/CLAM/patching_results"

# 4. Esegui lo script di segmentazione (SENZA --patch e --stitch)
# NOTA: se stai usando resezioni invece di biopsie, cambia --preset in bwh_resection.csv
python create_patches_fp.py \
    --source $WSI_DIR \
    --save_dir $OUTPUT_DIR \
    --patch_size 256 \
    --seg \
    --preset tcga.csv

echo "--- Job completato ---"
echo "Data e ora: $(date)"