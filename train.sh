# Will create pytroch model checkpoints "*.pth" and "log.txt" files in "./output/"
# calculates ADD, ADD-S, ADD(-S), avrg. t, and avrg R scores in "eval_test_backbone"


# Resume checkpoint training with mask_rcnn backbone # ~20s/it
python main.py --eval_batch_size 32 --batch_size 32 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --backbone maskrcnn --backbone_cfg ./configs/ycbv_rcnn.yaml \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output/ --bbox_mode backbone --epochs 50

# ~10s/it
 python main.py --eval_batch_size 16 --batch_size 16 --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_maskrcnn.pth --backbone maskrcnn --backbone_cfg ./configs/ycbv_rcnn.yaml \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output/ --bbox_mode backbone --epochs 50


# Resume checkpoint training with YOLO backbone
python main.py --eval_batch_size 32 --batch_size 32 --enc_layers 5 --dec_layers 5 --lr 0.00002 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_yolo.pth --backbone yolov4 --backbone_cfg ./configs/ycbv_yolov4-csp.cfg 
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json 
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output/ --bbox_mode backbone --epochs 50 --translation_loss_coef 2 


# Train PoET from scratch on YOLO backbone # ~13s/it
 python main.py --enc_layers 5 --dec_layers 5 --lr 0.00002 --nheads 16 --backbone yolov4 --lr_backbone 0.0 --backbone_cfg ./configs/ycbv_yolov4-csp.cfg \
 --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --dataset_path  /media/sebastian/TEMP/poet/datasets/ycbv  --class_info /annotations/ycbv_classes.json \
 --model_symmetry /annotations/ycbv_symmetries.json --train_set train_pbr --output_dir output/ --bbox_mode backbone --epochs 50 --translation_loss_coef 2 