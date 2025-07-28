#!/bin/bash

# Isaac Sim Dataset Training Script for PoET
# This script trains the PoET model on Isaac Sim dataset

echo "Starting PoET training on Isaac Sim dataset..."

# Initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the poet_isaac environment
conda activate poet_isaac

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=1

# Training parameters
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_training"
DATASET_PATH="../isaac_sim_poet_dataset_force_new"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 21 \
    --lr 2e-5 \
    --lr_backbone 0.0 \
    --batch_size 8 \
    --epochs 50 \
    --rgb_augmentation \
    --bbox_mode gt \
    --jitter_probability 0.5 \
    --backbone yolov4 \
    --backbone_cfg ./configs/ycbv_yolov4-csp.cfg \
    --backbone_weights ./weights/poet_ycbv_yolo.pth \
    --position_embedding sine \
    --nheads 16 \
    --enc_layers 5 \
    --dec_layers 5 \
    --translation_loss_coef 5 \
    --rotation_loss_coef 3 \
    --force_loss_coef 2.0 \
    --force_symmetry_coef 0.5 \
    --force_consistency_coef 0.5 \
    --force_scale_factor 5.0 \
    --use_force_prediction \
    --use_graph_transformer \
    --graph_hidden_dim 256 \
    --graph_num_layers 4 \
    --graph_num_heads 4 \
    --eval_interval 1 \
    --eval_by_epoch \
    --save_interval 5 \
    --device cuda \
    --class_info /annotations/classes.json \
    --model_symmetry /annotations/isaac_sim_symmetries.json > train_yolo.log 2>&1 &

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "To resume training, add --resume $OUTPUT_DIR/checkpoint.pth to the command above"