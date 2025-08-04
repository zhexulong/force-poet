# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import data_utils.samplers as samplers
from data_utils import build_dataset
from engine import train_one_epoch, train_one_epoch_with_iter_eval, pose_evaluate, bop_evaluate
from models import build_model
from evaluation_tools.pose_evaluator_init import build_pose_evaluator
from inference_tools.inference_engine import inference
from tabulate import tabulate

import util.logger
from util.logger import warn, err
from CorrectedSummaryWriter import CorrectedSummaryWriter

def get_args_parser():

    parser = argparse.ArgumentParser('Pose Estimation Transformer', add_help=False)

    # Learning
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=5.0, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='yolov4', type=str, choices=['yolov4', 'maskrcnn', 'fasterrcnn', 'dinorcnn', 'dinoyolo'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_cfg', default='configs/ycbv_yolov4-csp.cfg', type=str,
                        help="Path to the backbone config file to use")
    parser.add_argument('--backbone_weights', default=None, type=str,
                        help="Path to the pretrained weights for the backbone."
                             "None if no weights should be loaded.")
    parser.add_argument('--backbone_conf_thresh', default=0.4, type=float,
                        help="Backbone confidence threshold which objects to keep.")
    parser.add_argument('--backbone_iou_thresh', default=0.5, type=float, help="Backbone IOU threshold for NMS")
    parser.add_argument('--backbone_agnostic_nms', action='store_true',
                        help="Whether backbone NMS should be performed class-agnostic")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * DINO BACKBONE
    parser.add_argument('--dino_caption', default=None, type=str, help='Caption for Grounding DINO object detection')
    parser.add_argument('--dino_args', default="models/groundingdino/config/GroundingDINO_SwinT_OGC.py", type=str, help='Args for Grounding DINO backbone')
    parser.add_argument('--dino_checkpoint', default="models/groundingdino/weights/groundingdino_swint_ogc.pth", type=str, help='Checkpoint for Grounding DINO backbone')
    parser.add_argument('--dino_box_threshold', default=0.35, type=float, help='Bounding Box threshold for Grounding DINO')
    parser.add_argument('--dino_txt_threshold', default=0.25, type=float, help='Text threshold for Grounding DINO')
    parser.add_argument('--dino_cos_sim', default=0.9, type=float, help='Cosine similarity for matching Grounding DINO predictions to labels')
    parser.add_argument('--dino_bbox_viz', default=False, type=bool, help='Visualize Grounding DINO bounding box predictions and labels')

    # ** PoET configs
    parser.add_argument('--bbox_mode', default='gt', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--query_embedding', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer query embeddings are learned or determined by the bounding boxes')
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'quat', 'silho_quat'),
                        help="Determine the rotation representation with which PoET is trained.")
    parser.add_argument('--class_mode', default='specific', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # * Graph Transformer and Force Prediction
    parser.add_argument('--use_graph_transformer', action='store_true', default=True,
                        help="Whether to use graph transformer for force prediction")
    parser.add_argument('--graph_hidden_dim', default=None, type=int,
                        help="Hidden dimension for graph transformer (default: same as hidden_dim)")
    parser.add_argument('--graph_num_layers', default=2, type=int,
                        help="Number of layers in graph transformer")
    parser.add_argument('--graph_num_heads', default=8, type=int,
                        help="Number of attention heads in graph transformer")
    parser.add_argument('--use_force_prediction', action='store_true', default=False,
                        help="Whether to predict forces")

    # * Matcher
    parser.add_argument('--matcher_type', default='pose', choices=['pose'], type=str)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    # Pose Estimation losses
    parser.add_argument('--translation_loss_coef', default=1, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rotation_loss_coef', default=2.0, type=float, help='Loss weighing parameter for the rotation')
    parser.add_argument('--force_loss_coef', default=2.0, type=float, help='Loss weighing parameter for the force prediction')
    parser.add_argument('--force_symmetry_coef', default=1.0, type=float, help='Loss weighing parameter for force symmetry constraint (Newton\'s 3rd law)')
    parser.add_argument('--force_consistency_coef', default=1.0, type=float, help='Loss weighing parameter for force consistency constraint (Newton\'s 1st law)')
    parser.add_argument('--hard_negative_ratio', default=0.2, type=float, 
                        help='Ratio of hard negative samples to keep for improved force loss training (0.1-1.0)')
    parser.add_argument('--force_scale_factor', default=5.0, type=float,
                        help='Scaling factor for force values during training to improve numerical stability (default: 5.0)')

    # dataset parameters
    parser.add_argument('--dataset', default='ycbv', type=str, choices=('ycbv', 'lmo', 'icmi', 'custom'),
                        help="Choose the dataset to train/evaluate PoET on.")
    parser.add_argument('--dataset_path', default='/data', type=str,
                        help='Path to the dataset ')
    parser.add_argument('--train_set', default="train", type=str, help="Determine on which dataset split to train")
    parser.add_argument('--eval_set', default="test", type=str, help="Determine on which dataset split to evaluate")
    parser.add_argument('--test_set', default="test", type=str, help="Determine on which dataset split to test")
    parser.add_argument('--synt_background', default=None, type=str,
                        help="Directory containing the background images from which to sample")
    parser.add_argument('--n_classes', default=21, type=int, help="Number of classes present in the dataset")
    parser.add_argument('--jitter_probability', default=0.5, type=float,
                        help='If bbox_mode is set to jitter, this value indicates the probability '
                             'that jitter is applied to a bounding box.')
    parser.add_argument('--rgb_augmentation', action='store_true',
                        help='Activate image augmentation for training pose estimation.')
    parser.add_argument('--grayscale', action='store_true', help='Activate grayscale augmentation.')

    # * Evaluator
    parser.add_argument('--eval_interval', type=int, default=500,
                        help="Iteration interval after which the current model is evaluated")
    parser.add_argument('--eval_by_epoch', action='store_true',
                        help="Use epoch-based evaluation instead of iteration-based evaluation")
    parser.add_argument('--class_info', type=str, default='/annotations/classes.json',
                        help='path to .txt-file containing the class names')
    parser.add_argument('--models', type=str, default='/models_eval/',
                        help='path to a directory containing the classes models')
    parser.add_argument('--model_symmetry', type=str, default='/annotations/symmetries.json',
                        help='path to .json-file containing the class symmetries')

    # * Inference
    parser.add_argument('--inference', action='store_true',
                        help="Flag indicating that PoET should be launched in inference mode.")
    parser.add_argument('--inference_path', type=str,
                        help="Path to the directory containing the files for inference.")
    parser.add_argument('--inference_output', type=str,
                        help="Path to the directory where the inference results should be stored.")

    # * Misc
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--save_interval', default=5, type=int,
                        help="Epoch interval after which the current checkpoint will be stored")
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode')
    parser.add_argument('--eval_bop', action='store_true', help="Run model in BOP challenge evaluation mode")
    parser.add_argument('--test', action='store_true', help="Run model in BOP challenge test mode")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # * Distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch ')
    parser.add_argument('--world_size', default=3, type=int,
                        help='number of distributed processes/ GPUs to use')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend') 
    parser.add_argument('--local_rank', default=0, type=int,
                        help='rank of the process')     
    parser.add_argument('--gpu', default=0, type=int, help='rank of the process')

    return parser

def test(pose_evaluator, model, matcher, args, device, output_dir: Path, epoch: int = None):
    if args.test_set is not None:
        print('Start testing...')
        test_start_time = time.time()

        dataset_test = build_dataset(image_set=args.test_set, args=args)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(dataset_test, args.eval_batch_size, sampler=sampler_test,
                                      drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                      pin_memory=True)

        eval_results = pose_evaluate(model, matcher, pose_evaluator, data_loader_test, args.test_set,
                                                   args.bbox_mode,
                                                   args.rotation_representation, device, str(output_dir), args, epoch)
        
        # Extract the main metrics for backward compatibility
        avg_trans_err = eval_results['avg_trans']
        avg_rot_err = eval_results['avg_rot']

        test_total_time = time.time() - test_start_time
        test_total_time_str = str(datetime.timedelta(seconds=int(test_total_time)))
        print('Testing time {}'.format(test_total_time_str))

        return avg_trans_err, avg_rot_err, test_total_time
    else:
        print("Cannot test model because args.test_set is None! Skipping testing ...")
        return None, None, None


def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model and evaluator
    model, criterion, matcher = build_model(args)
    model.to(device)


    pose_evaluator = build_pose_evaluator(args)

    model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # Build the dataset for training and validation
    dataset_train = build_dataset(image_set=args.train_set, args=args)
    dataset_val = build_dataset(image_set=args.eval_set, args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    
    data_loader_val = DataLoader(dataset_val, args.eval_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, args.gamma)

    if args.distributed:
        print(f'\nUsing DistributedDataParallel\n')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    headers = ["Argument", "Value", "Description"]
    data = [
      ["Flags", ""],
      ["Inference", str(args.inference)],
      ["Eval", str(args.eval)],
      ["Eval BOP", str(args.eval_bop)],
      ["Distributed", str(args.distributed)],
      ["RGB Augm.", str(args.rgb_augmentation), "Whether to augment training images with RGB transformations."],
      ["Grayscale Augm.", str(args.grayscale), "Whether to augment training images with Grayscale transformations."],
      ["", ""],
      ["Architecture", ""],
      ["Enc. Layers", str(args.enc_layers)],
      ["Dec. Layers", str(args.dec_layers)],
      ["Num. Heads", str(args.nheads)],
      ["Num. Object Queries", str(args.num_queries), "Number of object queries per image. (Numb. of objects hypothesises per image)"],
      ["", ""],
      ["Resume", str(args.resume), "Model checkpoint to resume training of."],
      ["Backbone", str(args.backbone)],
      ["BBox Mode", str(args.bbox_mode)],
      ["Dataset", str(args.dataset)],
      ["Dataset Path", str(args.dataset_path)],
      ["N Classes", str(args.n_classes), "Number of total classes/labels."],
      ["Class Mode", str(args.class_mode)],
      ["Rot. Reprs.", str(args.rotation_representation)],
      ["", ""],
      ["Training", ""],
      ["Train Set", str(args.train_set)],
      ["Batch Size", str(args.batch_size)],
      ["Epochs", str(args.epochs)],
      ["Learning Rate", str(args.lr)],
      ["LR. Drop", str(args.lr_drop), "Decays learning rate all 'LR. Drop' epochs by multiplicative of 'Gamma'"],
      ["Gamma", str(args.gamma), "Multiplicative factor of learning rate drop"],
      ["Transl. Loss Coef.", str(args.translation_loss_coef), "Weighting of translation loss."],
      ["Rot. Loss Coef.", str(args.rotation_loss_coef), "Weighting of rotation loss."],
      ["Force Loss Coef.", str(args.force_loss_coef), "Weighting of force prediction loss."],
      ["Force Symmetry Coef.", str(args.force_symmetry_coef), "Weighting of force symmetry constraint (Newton's 3rd law)."],
      ["Force Consistency Coef.", str(args.force_consistency_coef), "Weighting of force consistency constraint (Newton's 1st law)."],
      ["Hard Negative Ratio", str(args.hard_negative_ratio), "Ratio of hard negative samples for improved force loss training."],
      ["Use Force Prediction", str(args.use_force_prediction), "Whether to enable force prediction."],
      ["Use Graph Transformer", str(args.use_graph_transformer), "Whether to use graph transformer for force prediction."],
      ["", ""],
      ["Eval", ""],
      ["Eval Batch Size", str(args.eval_batch_size)],
      ["Eval Set", str(args.eval_set)],
      ["", ""],
      ["Test", ""],
      ["Test Set", str(args.test_set)],
      ["", ""],
      ["Inference", ""],
      ["Inference Path", str(args.inference_path)],
      ["Inference Output", str(args.inference_output)],
    ]

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(tabulate(data, headers=headers, tablefmt="rounded_outline"))
    print("")
    print('Number of params:', n_parameters)
    print("")

    # Prepare output directory path
    output_dir = Path(args.output_dir)
    if "train" in args.output_dir or "tune" in args.output_dir:
      output_dir = Path(os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Load checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Filter out force_matrix_head parameters to allow normal initialization
        filtered_state_dict = {}
        force_matrix_keys = []
        for key, value in checkpoint['model'].items():
            if 'force_matrix_head' in key:
                force_matrix_keys.append(key)
                print(f"Skipping force_matrix_head parameter: {key}")
            else:
                filtered_state_dict[key] = value
        
        if force_matrix_keys:
            print(f"Excluded {len(force_matrix_keys)} force_matrix_head parameters from checkpoint loading.")
            print("force_matrix_head will use normal initialization.")
        
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(filtered_state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            warn(f"There are {len(missing_keys)} missing keys in state_dict!")
            # print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn(f"There are {len(unexpected_keys)} unexpected keys in state_dict!")
            # print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                # print(optimizer.param_groups)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                print("Successfully loaded optimizer and lr_scheduler from checkpoint.")
            except ValueError as e:
                if "doesn't match the size of optimizer's group" in str(e):
                    print(f"Warning: Cannot load optimizer state due to parameter size mismatch: {e}")
                    print("This is expected when force_matrix_head parameters are excluded from loading.")
                    print("Optimizer will use fresh initialization with current learning rate settings.")
                    # Still try to load lr_scheduler if possible
                    try:
                        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                        print("Successfully loaded lr_scheduler from checkpoint.")
                    except Exception as lr_e:
                        print(f"Warning: Cannot load lr_scheduler: {lr_e}")
                        print("lr_scheduler will use fresh initialization.")
                else:
                    # Re-raise if it's a different error
                    raise e

            # Fallback if gamma was an array previously
            if isinstance(lr_scheduler.gamma, list):
              lr_scheduler.gamma = lr_scheduler.gamma[0]

            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler
            #  (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    # Evaluate the models performance
    eval_epoch = None
    if args.eval:
        if args.resume:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None

        pose_evaluator.training = False
        pose_evaluator.testing = False
        pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
                      args.rotation_representation, device, args.output_dir, args, eval_epoch)
        return

    # Evaluate the model for the BOP challenge
    if args.eval_bop:
        print(args.dataset)
        pose_evaluator.training = False
        pose_evaluator.testing = False
        bop_evaluate(model, matcher, data_loader_val, args.eval_set, args.bbox_mode,
                     args.rotation_representation, device, args, args.output_dir)
        return

    if args.test:
        if args.resume:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None
        pose_evaluator.training = False
        pose_evaluator.testing = False
        avg_trans_err, avg_rot_err, test_total_time_str = test(pose_evaluator, model, matcher, args, device, output_dir, eval_epoch)
        return

    print("Start training")
    util.logger.saveArgs(output_dir, args)

    start_time = time.time()
    writer = CorrectedSummaryWriter(os.path.join(output_dir))
    pose_evaluator.writer = writer
    pose_evaluator.training = True
    pose_evaluator.testing = False

    epoch = 0
    global_iteration = 0
    try:
        best_loss = sys.float_info.max
        
        if args.eval_by_epoch:
            # Original epoch-based evaluation
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                
                # Set epoch for curriculum learning in criterion
                criterion.set_epoch(epoch)

                start = time.time()
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
                stop = time.time()

                lr_scheduler.step()
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'checkpoint_latest.pth']
                    # extra checkpoint before LR drop and every save_interval epochs
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_interval == 0:
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)

                writer.add_scalar('Train/lr', train_stats["lr"], epoch)
                writer.add_scalar('Train/loss', train_stats["loss"], epoch)
                writer.add_scalar('Train/position_loss', train_stats["position_loss"], epoch)
                writer.add_scalar('Train/rotation_loss', train_stats["rotation_loss"], epoch)
                writer.add_scalar('Train/times/time_per_epoch', stop - start, epoch)

                # Do evaluation on the validation set every n epochs
                if epoch % args.eval_interval == 0:
                    eval_results = pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
                                  args.rotation_representation, device, str(output_dir), None, epoch)
                    
                    # Extract the main metrics for backward compatibility
                    avg_trans_err = eval_results['avg_trans']
                    avg_rot_err = eval_results['avg_rot']

                    # Save model if best translation and rotation result
                    if args.output_dir:
                        checkpoint_loss = (avg_trans_err + avg_rot_err) / 2
                        if checkpoint_loss < best_loss:
                            best_loss = checkpoint_loss
                            utils.save_on_master({
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': args,
                            }, output_dir / 'checkpoint.pth')

                    writer.add_scalar("Val/avg_trans_err", avg_trans_err, epoch)
                    writer.add_scalar("Val/avg_rot_err", avg_rot_err, epoch)
                    writer.add_scalar("Val/avg_err", (avg_trans_err + avg_rot_err) / 2, epoch)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 'epoch': epoch,
                                 'n_parameters': n_parameters}
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 'epoch': epoch,
                                 'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
        else:
            # New iteration-based evaluation
            def eval_func_wrapper():
                """评估函数包装器"""
                eval_results = pose_evaluate(
                    model, matcher, pose_evaluator, data_loader_val, args.eval_set, 
                    args.bbox_mode, args.rotation_representation, device, str(output_dir), None, None
                )
                
                # Extract the main metrics
                avg_trans_err = eval_results['avg_trans']
                avg_rot_err = eval_results['avg_rot']
                
                # Save model if best translation and rotation result
                nonlocal best_loss
                if args.output_dir:
                    checkpoint_loss = (avg_trans_err + avg_rot_err) / 2
                    if checkpoint_loss < best_loss:
                        best_loss = checkpoint_loss
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, output_dir / 'checkpoint.pth')
                
                writer.add_scalar("Val/avg_trans_err", avg_trans_err, global_iteration)
                writer.add_scalar("Val/avg_rot_err", avg_rot_err, global_iteration)
                writer.add_scalar("Val/avg_err", (avg_trans_err + avg_rot_err) / 2, global_iteration)
                
                print(f"评估结果 - 平移误差: {avg_trans_err:.4f}, 旋转误差: {avg_rot_err:.4f}")
                return avg_trans_err, avg_rot_err
            
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                
                # Set epoch for curriculum learning in criterion
                criterion.set_epoch(epoch)

                start = time.time()
                train_stats, global_iteration = train_one_epoch_with_iter_eval(
                    model, criterion, data_loader_train, optimizer, device, epoch, 
                    global_iteration, args.eval_interval, eval_func_wrapper, {},
                    args.clip_max_norm)
                stop = time.time()

                lr_scheduler.step()
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'checkpoint_latest.pth']
                    # extra checkpoint before LR drop and every save_interval epochs
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_interval == 0:
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)

                writer.add_scalar('Train/lr', train_stats["lr"], epoch)
                writer.add_scalar('Train/loss', train_stats["loss"], epoch)
                writer.add_scalar('Train/position_loss', train_stats["position_loss"], epoch)
                writer.add_scalar('Train/rotation_loss', train_stats["rotation_loss"], epoch)
                writer.add_scalar('Train/times/time_per_epoch', stop - start, epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

    except KeyboardInterrupt as e:
        warn(f"Keyboard Interrupt caught!")
        warn(f"Logging hyperparameters and doing a final test run ...")
        pass
    except Exception as e:
        err(f"Exception during training: {e}")
        traceback.print_exc()
        err("Exiting program ...")

    pose_evaluator.training = True
    pose_evaluator.testing = True
    avg_trans_err, avg_rot_err, test_total_time_str = test(pose_evaluator, model, matcher, args, device, output_dir, epoch)

    writer.add_scalar("Test/avg_trans_err", avg_trans_err, epoch)
    writer.add_scalar("Test/avg_rot_err", avg_rot_err, epoch)

    # --------------------
    # Log Hyperparameters
    writer.add_hparams(
      {
          "Batch Size": args.eval_batch_size,
          "Eval Batch Size": args.eval_batch_size,
          "Learning Rate": args.lr,
          "Transl. Loss Coef.": args.translation_loss_coef,
          "Rot. Loss Coef.": args.rotation_loss_coef,
          "Enc. Layers": args.enc_layers,
          "Dec. Layers": args.dec_layers,
          "Number Heads": args.nheads,
          "Number Object Queries": args.num_queries,
          "RGB Augmentation": args.rgb_augmentation,
          "Grayscale Augmentation": args.grayscale,
      },
      {
          "Test/avg_rot_err": avg_rot_err,
          "Test/avg_trans_err": avg_trans_err,
      }
    )

    writer.close()

    # --------------------------
    # Log training/test times
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # Log execution times to file
    if args.output_dir and utils.is_main_process():
      with (output_dir / "log.txt").open("a") as f:
        obj = {
          "training_time": total_time_str,
          "test_total_time": test_total_time_str,
        }
        f.write(json.dumps(obj) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.inference:
        args.bbox_mode = "backbone"
        inference(args)
    else:
        main(args)
