# python main.py --eval_batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --eval_bop --backbone maskrcnn --backbone_cfg ./configs/ycbv_rcnn.yaml --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output/


# BOP EVALUATION
#
# Creates /output/bop_[bbox_mode]
# Will create "ycbv.csv" for bop evaluation
python main.py --eval_batch_size 32 --batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --eval_bop --backbone maskrcnn --backbone_cfg ./configs/ycbv_rcnn.yaml \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output_rcnn/ --bbox_mode backbone --eval_set test_all


# ------ EVALUATION
# Creates /output/eval_[image_set]_[bbox_mode]
python main.py --eval_batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --eval --backbone maskrcnn --backbone_cfg ./configs/ycbv_rcnn.yaml \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output_rcnn/ --bbox_mode backbone --eval_set test_all


# ---- EVALUATION WITH YOLOv4 BACKBONE
# Creates /output/eval_[image_set]_[bbox_mode]_[epoch_checkpoint]
python main.py --eval_batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_yolo.pth --eval --backbone yolov4 --backbone_cfg ./configs/ycbv_yolov4-csp.cfg \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output_new/ --bbox_mode backbone --eval_set test_all --lr_backbone 0.0 --translation_loss_coef 2 --rgb_augmentation --grayscale


# ---- EVALUATION WITH **DINO - RCNN** BACKBONE
# Creates /output/eval_[image_set]_[bbox_mode]_[epoch_checkpoint]
python main.py --eval_batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --eval --backbone dinorcnn --backbone_cfg ./configs/ycbv_rcnn.yaml \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output_dino/ --bbox_mode backbone --eval_set test_all


# ---- EVALUATION WITH **DINO - YOLOv4** BACKBONE
python main.py --eval_batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_yolo.pth --eval --backbone dinoyolo --backbone_cfg ./configs/ycbv_yolov4-csp.cfg \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --dataset_path /media/sebastian/TEMP/poet/datasets/ycbv --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output_dino_yolo_batch/ --bbox_mode backbone --eval_set test_all --lr_backbone 0.0


# -------------- BOP TOOLKIT VISUALIZATION:
# Copy ycbv.csv results file to TEMP/poet/results
# Rename to [bbox_mode]_ycbv-[test_set], e.g.: backbone_ycbv-test_all.csv
# execute: vis_est_poses.py from bop_toolkit
# - this looks for backbone_ycbv-test_all.csv
# - and creates /TEMP/poet/output/... 