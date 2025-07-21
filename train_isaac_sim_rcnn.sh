#!/bin/bash

# Isaac Sim Dataset Training Script for PoET
# This script trains the PoET model on Isaac Sim dataset

echo "Starting PoET training on Isaac Sim dataset..."

# Initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the poet_isaac environment
conda activate poet_isaac

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Training parameters
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_training"
DATASET_PATH="../isaac_sim_poet_dataset"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
nohup python main.py \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 22 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --batch_size 8 \
    --epochs 50 \
    --rgb_augmentation \
    --bbox_mode gt \
    --jitter_probability 0.5 \
    --backbone maskrcnn \
    --backbone_cfg ./configs/ycbv_rcnn.yaml \
    --backbone_weights ./weights/ycbv_maskrcnn_checkpoint.pth.tar \
    --position_embedding sine \
    --nheads 16 \
    --enc_layers 5 \
    --dec_layers 5 \
    --translation_loss_coef 2 \
    --rotation_loss_coef 1 \
    --eval_interval 1 \
    --save_interval 5 \
    --device cuda \
    --class_info /annotations/classes.json \
    --model_symmetry /annotations/isaac_sim_symmetries.json > train.log 2>&1 &

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "To resume training, add --resume $OUTPUT_DIR/checkpoint.pth to the command above"