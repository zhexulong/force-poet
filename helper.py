import numpy as np


class Args:
  def __init__(self):
    # Learning rate and optimization parameters
    self.lr = 2e-4
    self.lr_backbone_names = ["backbone.0"]
    self.lr_backbone = 2e-5
    self.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    self.lr_linear_proj_mult = 0.1
    self.batch_size = 16
    self.eval_batch_size = 16
    self.weight_decay = 1e-4
    self.epochs = 50
    self.lr_drop = 100
    self.gamma = 0.1
    self.clip_max_norm = 0.1

    # Backbone parameters
    self.backbone = 'yolov4'
    self.backbone_cfg = '../configs/ycbv_yolov4-csp.cfg'
    self.backbone_weights = None
    self.backbone_conf_thresh = 0.4
    self.backbone_iou_thresh = 0.5
    self.backbone_agnostic_nms = False
    self.dilation = False
    self.position_embedding = 'sine'
    self.position_embedding_scale = 2 * np.pi
    self.num_feature_levels = 4

    # PoET configs
    self.bbox_mode = 'gt'
    self.reference_points = 'bbox'
    self.query_embedding = 'bbox'
    self.rotation_representation = '6d'
    self.class_mode = 'specific'

    # Transformer parameters
    self.enc_layers = 6
    self.dec_layers = 6
    self.dim_feedforward = 1024
    self.hidden_dim = 256
    self.dropout = 0.1
    self.nheads = 8
    self.num_queries = 10
    self.dec_n_points = 4
    self.enc_n_points = 4

    # Matcher parameters
    self.matcher_type = 'pose'
    self.set_cost_class = 1.0
    self.set_cost_bbox = 1.0
    self.set_cost_giou = 2.0

    # Loss coefficients
    self.aux_loss = True
    self.translation_loss_coef = 1.0
    self.rotation_loss_coef = 1.0

    # Dataset parameters
    self.dataset = 'ycbv'
    self.dataset_path = '/data'
    self.train_set = 'train'
    self.eval_set = 'test'
    self.test_set = 'test'
    self.synt_background = None
    self.n_classes = 21
    self.jitter_probability = 0.5
    self.rgb_augmentation = False
    self.grayscale = False

    # Evaluator parameters
    self.eval_interval = 10
    self.class_info = '/annotations/classes.json'
    self.models = '/models_eval/'
    self.model_symmetry = '/annotations/symmetries.json'

    # Inference parameters
    self.inference = False
    self.inference_path = None
    self.inference_output = None

    # Misc parameters
    self.sgd = False
    self.save_interval = 5
    self.output_dir = ''
    self.device = 'cuda'
    self.seed = 42
    self.resume = ''
    self.start_epoch = 0
    self.eval = False
    self.eval_bop = False
    self.test = False
    self.num_workers = 0
    self.cache_mode = False

    # Distributed training parameters
    self.distributed = False
    self.world_size = 3
    self.dist_url = 'env://'
    self.dist_backend = 'nccl'
    self.local_rank = 0
    self.gpu = 0


controls = {
    # movement in directions
    'w': 'forward',
    's': 'backward',
    'a': 'left',
    'd': 'right',

    # ascend/descent slowly
    'space': 'up',
    'left shift': 'down',
    'right shift': 'down',

    # yaw slowly
    'q': 'counter_clockwise',
    'e': 'clockwise',

    # arrow keys for fast turns and altitude adjustments
    'left': lambda drone, speed: drone.counter_clockwise(speed*2),
    'right': lambda drone, speed: drone.clockwise(speed*2),
    'up': lambda drone, speed: drone.up(speed*2),
    'down': lambda drone, speed: drone.down(speed*2),

    # lift off and land
    'x': lambda drone, speed: drone.takeoff(),
    'y': lambda drone, speed: drone.land(),
}