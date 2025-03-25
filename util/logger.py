from argparse import Namespace
from pathlib import Path

from tabulate import tabulate

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m' # orange on some systems
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
LIGHT_GRAY = '\033[37m'
DARK_GRAY = '\033[90m'
BRIGHT_RED = '\033[91m'
BRIGHT_GREEN = '\033[92m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_CYAN = '\033[96m'
WHITE = '\033[97m'

RESET = '\033[0m' # called to return to standard terminal text color


def info(msg):
  print(f"[INFO] {msg}")

def succ(msg):
  print(f"{GREEN}[SUCC]{RESET} {msg}")

def warn(msg):
  print(f"{YELLOW}[WARN]{RESET} {msg}")

def err(msg):
  print(f"{RED}[INFO]{RESET} {msg}")

def saveArgs(output_dir: Path, args: Namespace):
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
    ["Num. Object Queries", str(args.num_queries),
     "Number of object queries per image. (Numb. of objects hypothesises per image)"],
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
    ["DINO Caption", str(args.dino_caption)],
    ["", ""],
    ["Training", ""],
    ["Train Set", str(args.train_set)],
    ["Batch Size", str(args.batch_size)],
    ["Epochs", str(args.epochs)],
    ["Learning Rate", str(args.lr)],
    ["LR. Drop", str(args.lr_drop), "Decays learning rate all 'LR. Drop' epochs by multiplicative of 'Gamma'"],
    ["Gamma", str(args.gamma), "Multiplicative factor of learning rate drop"],
    ["Transl. Loss Coef.", str(args.translation_loss_coef), "Weighting of translation loss."],
    ["Rot. Loss Coef.", str(args.rotation_loss_coef)],
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

  with open(Path(output_dir, "args.txt"), "w") as f:
    f.write(tabulate(data, headers=headers, tablefmt="rounded_outline"))