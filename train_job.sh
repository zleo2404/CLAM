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
# La cartella dei risultati e' results/<exp_code>_<model_type>_<model_size>_s<seed>
# es. results/HER2_20X_clam_sb_small_s1
FEAT_DIR="features_resnetNorm_20x"
SPLIT_DIR="task_1_tumor_vs_normal_100_kfold"
SEED=42


CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out 0.25 \
    --max_epochs 120 \
    --early_stopping \
    --lr 1e-4 \
    --reg 1e-4 \
    --scheduler cosine \
    --scheduler_min_lr 1e-6 \
    --opt adamw \
    --k 5 \
    --exp_code HER2_20X_ResNetNormT \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss ce \
    --bag_weight 0.7 \
    --B 8 \
    --task task_1_tumor_vs_normal \
    --model_type clam_sb \
    --model_size small \
    --log_data \
    --data_root_dir /scratch.hpc/leonardo.meloni/CLAM \
    --feat_dir $FEAT_DIR \
    --split_dir $SPLIT_DIR \
    --embed_dim 1024 \
    --seed 42

echo "--- Addestramento completato ---"
echo "Data e ora: $(date)"