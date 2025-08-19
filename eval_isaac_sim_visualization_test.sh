#!/bin/bash

# Isaac Sim 数据集可视化测试脚本
# 测试力图和6D姿态可视化功能

echo "Starting Isaac Sim visualization test..."

# 设置参数
CONFIG_FILE="configs/isaac_sim.yaml"
OUTPUT_DIR="./results/isaac_sim_visualization_test"
DATASET_PATH="../isaac_sim_poet_dataset_force_point"
# CHECKPOINT_PATH="./results/isaac_sim_training/2025-08-17_02:13:05/checkpoint0079.pth"
# CHECKPOINT_PATH="./results/isaac_sim_training/2025-08-18_01:44:47/checkpoint0074.pth"
CHECKPOINT_PATH="./results/isaac_sim_training/2025-08-18_22:23:19/checkpoint0074.pth"
CLASS_INFO="/annotations/classes.json"
MODEL_SYMMETRY="/annotations/isaac_sim_symmetries.json"
VIZ_OUTPUT_DIR="$OUTPUT_DIR/visualizations"
MIN_THRESHOLD=0.5
MAX_THRESHOLD=0.5
THRESHOLD_STEP=0.05

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $VIZ_OUTPUT_DIR

echo "Running visualization test..."
echo "Dataset path: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Visualization directory: $VIZ_OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Testing with 10% visualization sampling rate"

# 运行可视化测试（使用10%采样率进行测试）
CUDA_VISIBLE_DEVICES=1 python eval_threshold_analysis.py \
    --dataset custom \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_classes 22 \
    --batch_size 2 \
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
    --model_symmetry $MODEL_SYMMETRY \
    --enable_visualization \
    --viz_sample_rate 0.1 \
    --viz_output_dir $VIZ_OUTPUT_DIR \
    --viz_force_scale 0.1 \
    --viz_min_force 0.01

echo ""
echo "Isaac Sim visualization test completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Visualizations saved to: $VIZ_OUTPUT_DIR"
echo ""
echo "Checking visualization outputs:"
ls -la $VIZ_OUTPUT_DIR/ | head -10