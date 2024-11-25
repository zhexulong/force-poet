python main.py --enc_layers 5 --dec_layers 5 --nheads 16 --resume ./output/checkpoint0039.pth --inference --inference_path /media/sebastian/TEMP/poet/test --inference_output ./output_inference/ \
 --backbone dinoyolo --backbone_cfg ./configs/ycbv_rcnn.yaml --backbone_weights /media/sebastian/TEMP/poet/ycbv_maskrcnn_checkpoint.pth.tar

 python main.py --enc_layers 5 --dec_layers 5 --nheads 16 --resume /media/sebastian/TEMP/poet/poet_ycbv_yolo.pth --inference --inference_path /media/sebastian/TEMP/poet/test --inference_output ./output_inference/ \
 --backbone dinoyolo --backbone_cfg ./configs/ycbv_yolov4-csp.cfg --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --dataset_path /media/sebastian/TEMP/poet/datasets/ycbv --class_info /annotations/ycbv_classes.json --lr_backbone 0.0

######################################################################################################

 pythn main.py --enc_layers 5 --dec_layers 5 --nheads 16 --resume /home/sebastian/repos/poet/train/2024-10-02_18:21:04/checkpoint.pth --inference --inference_path /home/sebastian/repos/master_project/dataset/records/cabinet_1/rgb \
 --inference_output inf/dino_yolo_custom/ --backbone dinoyolo --backbone_cfg ./configs/ycbv_yolov4-csp.cfg --backbone_weights /media/sebastian/TEMP/poet/ycbv_yolo_weights.pt --lr_backbone 0.0 --dataset_path /media/sebastian/TEMP/poet/datasets/custom \
 --class_info /annotations/custom_classes.json --model_symmetry /annotations/custom_symmetries.json --dataset custom --rgb_augmentation --grayscale --class_mode agnostic


######## DRONESPACE
python main.py --enc_layers 5 --dec_layers 5 --nheads 16 --resume /home/wngicg/Desktop/repos/poet/results/train/2024-10-06_12_31_12/checkpoint.pth --inference --inference_path /home/wngicg/Desktop/repos/datasets/custom/val/cabinet_1/rgb/ \
--inference_output inf/dino_yolo_custom/ --backbone dinoyolo --backbone_cfg ./configs/ycbv_yolov4-csp.cfg --backbone_weights /home/wngicg/Desktop/repos/ycbv_yolo_weights.pt --lr_backbone 0.0 --dataset_path /home/wngicg/Desktop/repos/datasets/custom \
--class_info /annotations/custom_classes.json --model_symmetry /annotations/custom_symmetries.json --dataset custom --rgb_augmentation --grayscale --class_mode agnostic


python main.py --enc_layers 5 --dec_layers 5 --nheads 16 --resume /home/wngicg/Desktop/repos/poet/results/train/2024-10-06_12_31_12/checkpoint.pth --inference --inference_path /home/wngicg/repos/poet/demo/fly/imgs_8/rgb \
--inference_output results/inf/dino_yolo_demo/ --backbone dinoyolo --backbone_cfg ./configs/ycbv_yolov4-csp.cfg --backbone_weights /home/wngicg/Desktop/repos/ycbv_yolo_weights.pt --lr_backbone 0.0 --dataset_path /home/wngicg/Desktop/repos/datasets/custom \
--class_info /annotations/custom_classes.json --model_symmetry /annotations/custom_symmetries.json --dataset custom --rgb_augmentation --grayscale --class_mode agnostic
