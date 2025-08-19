#!/bin/bash

# Isaac Sim 数据集阈值分析评估脚本
# 包含接触点分类和条件力回归评估

echo "Starting Isaac Sim threshold analysis evaluation..."

# 设置参数
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_threshold_analysis"
DATASET_PATH="../isaac_sim_poet_dataset_force_point"
# CHECKPOINT_PATH="./results/isaac_sim_training/2025-08-01_01:53:16/checkpoint0094.pth"
CHECKPOINT_PATH="./results/isaac_sim_training/2025-08-17_02:13:05/checkpoint0079.pth"
CLASS_INFO="/annotations/classes.json"
MODEL_SYMMETRY="/annotations/isaac_sim_symmetries.json"
MIN_THRESHOLD=0.05
MAX_THRESHOLD=0.9
THRESHOLD_STEP=0.05
# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "Running threshold analysis..."
echo "Dataset path: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Threshold range: $MIN_THRESHOLD to $MAX_THRESHOLD (step: $THRESHOLD_STEP)"

# 运行阈值分析评估
CUDA_VISIBLE_DEVICES=1 python eval_threshold_analysis.py \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 22 \
    --batch_size 4 \
    --eval_set val \
    --backbone maskrcnn \
    --backbone_cfg $CONFIG_FILE \
    --backbone_weights ./weights/ycbv_maskrcnn_checkpoint.pth.tar \
    --checkpoint_path $CHECKPOINT_PATH \
    --min_threshold $MIN_THRESHOLD \
    --max_threshold $MAX_THRESHOLD \
    --threshold_step $THRESHOLD_STEP \
    --nheads 16 \
    --enc_layers 5 \
    --dec_layers 5 \
    --graph_num_layers 4 \
    --graph_num_heads 8 \
    --device cuda \
    --class_info $CLASS_INFO \
    --model_symmetry $MODEL_SYMMETRY

echo "Isaac Sim threshold analysis evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
