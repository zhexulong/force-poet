#!/bin/bash

# Isaac Sim Dataset Training Script for PoET
# This script trains the PoET model on Isaac Sim dataset

echo "Starting PoET training on Isaac Sim dataset..."

# Initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the poet_isaac environment
conda activate poet_isaac

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=2

# Training parameters
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_training"
DATASET_PATH="../isaac_sim_poet_dataset_force_point"

# Create output directory
mkdir -p $OUTPUT_DIR

# Set checkpoint path for resuming training
# Note: Update this path to point to your actual checkpoint file
CHECKPOINT_PATH="./results/isaac_sim_training/2025-07-27_22:19:14/checkpoint0019.pth"

# Check if checkpoint exists, if not, look for alternatives
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Warning: Checkpoint file $CHECKPOINT_PATH not found!"
    echo "Looking for alternative checkpoint files..."
    
    # Try to find checkpoint_latest.pth
    LATEST_CHECKPOINT="./results/isaac_sim_training/2025-07-27_22:19:14/checkpoint_latest.pth"
    if [ -f "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
        echo "Found latest checkpoint: $CHECKPOINT_PATH"
    else
        # Try to find any checkpoint file in the directory
        FOUND_CHECKPOINT=$(find ./results/isaac_sim_training/2025-07-27_22:19:14/ -name "checkpoint*.pth" -type f | head -1)
        if [ -n "$FOUND_CHECKPOINT" ]; then
            CHECKPOINT_PATH="$FOUND_CHECKPOINT"
            echo "Found checkpoint: $CHECKPOINT_PATH"
        else
            echo "No checkpoint files found! Starting training from scratch."
            CHECKPOINT_PATH=""
        fi
    fi
fi

# Run training with force matrix prediction (resume from checkpoint if available)
CUDA_VISIBLE_DEVICES=2 nohup python main.py \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 22 \
    --lr 1e-4 \
    --lr_drop 95 \
    --lr_backbone 1e-6 \
    --batch_size 16 \
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
    --translation_loss_coef 5 \
    --rotation_loss_coef 3 \
    --eval_interval 1 \
    --eval_by_epoch \
    --save_interval 5 \
    --device cuda \
    --class_info /annotations/classes.json \
    --model_symmetry /annotations/isaac_sim_symmetries.json > train_rcnn_backbone.log 2>&1 &

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "To resume training, add --resume $OUTPUT_DIR/checkpoint.pth to the command above"