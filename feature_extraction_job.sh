#!/bin/bash
#SBATCH --job-name=HER2_FeatureExtraction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time=2-00:00:00
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

cd /scratch.hpc/leonardo.meloni/CLAM

WSI_DIR="/scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate"
H5_DIR="/scratch.hpc/leonardo.meloni/CLAM/patching_results_l1"
FEAT_DIR="/scratch.hpc/leonardo.meloni/CLAM/features_uni_l1_256"
CSV_FILE="/scratch.hpc/leonardo.meloni/CLAM/patching_results_l1/feature_extraction_list_uni_l1.csv"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TIMM_OFFLINE=1

export UNI_CKPT_PATH="/scratch.hpc/leonardo.meloni/conda_home/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
    --data_h5_dir $H5_DIR \
    --data_slide_dir $WSI_DIR \
    --csv_path $CSV_FILE \
    --feat_dir $FEAT_DIR \
    --batch_size 128 \
    --slide_ext .svs \
    --model_name uni_v1

echo "--- Job completato ---"
echo "Data e ora: $(date)"