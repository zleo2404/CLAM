#!/bin/bash
#SBATCH --job-name=HER2_FeatureExtraction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/leonardo.meloni/CLAM
#SBATCH --output=log_output_%j.txt
#SBATCH --error=log_error_%j.txt


echo "--- Inizio Addestramento CLAM-SB (HER2+ vs HER2-) ---"
echo "Data e ora: $(date)"

# Attiva l'ambiente virtuale
source /scratch.hpc/leonardo.meloni/clam_env/bin/activate

# Lancia l'addestramento
# Nota: exp_code darà il nome alla cartella dei risultati
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code HER2_UNI_CLAM_SB_100 \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task task_1_tumor_vs_normal \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /scratch.hpc/leonardo.meloni/CLAM \
    --embed_dim 1024

echo "--- Addestramento completato ---"
echo "Data e ora: $(date)"