#!/bin/bash

# Isaac Sim Dataset Evaluation Script for PoET
# This script evaluates the PoET model on Isaac Sim dataset

echo "Starting PoET evaluation on Isaac Sim dataset..."

# Initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the poet_isaac environment
conda activate poet_isaac

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=1

# Evaluation parameters
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_evaluation"
DATASET_PATH="../isaac_sim_poet_dataset_new"
CHECKPOINT_PATH="./weights/ycbv_maskrcnn_checkpoint.pth.tar"
CLASS_INFO="/annotations/classes.json"
MODEL_SYMMETRY="/annotations/isaac_sim_symmetries.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --eval \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 22 \
    --batch_size 8 \
    --rgb_augmentation \
    --bbox_mode gt \
    --backbone maskrcnn \
    --backbone_cfg $CONFIG_FILE \
    --backbone_weights $CHECKPOINT_PATH \
    --position_embedding sine \
    --nheads 16 \
    --enc_layers 5 \
    --dec_layers 5 \
    --translation_loss_coef 2 \
    --rotation_loss_coef 1 \
    --eval_set test_all \
    --device cuda \
    --class_info /annotations/classes.json \
    --model_symmetry /annotations/isaac_sim_symmetries.json > eval.log 2>&1 &

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
