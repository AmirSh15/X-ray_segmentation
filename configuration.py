import argparse
import os

import torch
import wandb
import yaml
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN
from detectron2.model_zoo import model_zoo
from yaml.loader import SafeLoader


def init_wab(
    wab_config_path,
    model_config,
    model_args,
    entity,
    project_name,
    key,
):
    import wandb

    if os.path.exists(wab_config_path):
        with open(wab_config_path, "r") as f:
            wab_config = list(yaml.load_all(f, Loader=SafeLoader))[0]
        wandb.login(key=wab_config["key"])
        wandb.init(
            project=project_name,
            config=dict(model_config),
            entity=wab_config["entity"],
        )
        wandb.config.update(model_args.__dict__)
    else:
        wandb.login(key=key)
        wandb.init(
            project=project_name,
            config=model_config,
            entity=entity,
        )


def detectron_config(ctgry_pre_address=""):
    parser = argparse.ArgumentParser(description="PyTorch X-ray Segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="mask_rcnn_R_101_C4_3x",
        help="model name",
        choices=[
            "mask_rcnn_R_101_C4_3x",
            "mask_rcnn_R_50_C4_1x",
            "mask_rcnn_R_50_DC5_1x",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CANDIDPTX",
        # default="ChestX-Det",
        help="dataset name",
        choices=["CANDIDPTX", "ChestX-Det"],
    )
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument(
        "--load_image_caption", type=bool, default=False, help="load image caption"
    )
    parser.add_argument(
        "--image_caption_model_add",
        type=str,
        default="../captioning_transformer/trained_models/backbone_model_mask_rcnn_R_101_C4_3x_checkpoint.pth",
        help="image caption model address",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--iter", type=int, default=30000, help="number of iterations in training"
    )
    parser.add_argument("--lr", type=float, default=0.00025, help="learning rate")
    parser.add_argument(
        "--lr_step_size",
        type=tuple,
        default=(3000, 5000, 7000),
        help="learning rate decrease step size",
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.1, help="learning rate gamma"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--max_patience", type=int, default=5, help="max patience for early stopping")
    parser.add_argument(
        "--validation_loss",
        type=list,
        default=["val"],
        help="the loss considered to early stopping",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="threshold for segmentation"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./trained_models", help="save directory"
    )
    parser.add_argument(
        "--periodic_hook_add",
        type=str,
        default="./Results",
        help="periodic hook address to save image instances",
    )
    parser.add_argument(
        "--periodic_hook_period", type=int, default=1000, help="periodic hook period"
    )
    parser.add_argument(
        "--infer_dir", type=str, default="./Results", help="inference directory"
    )
    parser.add_argument(
        "--infer_model_add",
        type=str,
        default="./trained_models/non_transferred_learning/CANDIDPTX_Dataset/mask_rcnn_R_101_C4_3x/512_val_loss/early_stopped_model.pth",
        help="inference model address",
    )
    parser.add_argument(
        "--data_path", type=str, default='./data', help='data path for captioning model (not used in segmentation project)'
    )
    parser.add_argument(
        "--enable_wab", type=bool, default=True, help="enable weight and bias"
    )
    parser.add_argument(
        "--wab_project", type=str, default=None, help="weight and bias project name"
    )
    parser.add_argument(
        "--wab_entity", type=str, default=None, help="weight and bias entity name"
    )
    parser.add_argument("--wab_key", type=str, default=None, help="weight and bias key")
    parser.add_argument(
        "--wab_config_path",
        type=str,
        default="../wandb_config.yaml",
        help="weight and bias config file path",
    )
    args = parser.parse_args()

    cfg = get_cfg()
    # cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg

    # specify the category names
    if args.dataset_name == "CANDIDPTX":
        # cfg.CATEGORIES = [{"supercategory": None, "id": 0, "name": "phnx"}]
        cfg.CATEGORIES = ["phnx"]
    elif args.dataset_name == "ChestX-Det":
        file = open(ctgry_pre_address + "data/ChestX_Det/annotations/Categories", "r")
        cfg.CATEGORIES = [e.strip("\n") for e in file.readlines()]

    # get configuration from model_zoo
    if args.model == "mask_rcnn_R_101_C4_3x":
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        )
    elif args.model == "mask_rcnn_R_50_C4_1x":
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        )
    elif args.model == "mask_rcnn_R_50_DC5_1x":
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
        )

    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.CATEGORIES)
    # cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # cfg.MODEL.BACKBONE.NAME = "build_retinanet_resnet_fpn_backbone"
    # cfg.MODEL.RESNETS.DEPTH = 34
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    # cfg.MODEL.RESNETS.RES5_DILATION = 1
    # cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
    # cfg.MODEL.BBOX_LOSS_TYPE = 'diou'
    # cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']

    # Solver
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.SOLVER.STEPS = args.lr_step_size
    cfg.SOLVER.GAMMA = args.lr_gamma
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.MOMENTUM = args.momentum
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.TEST.EVAL_PERIOD = 100
    cfg.MAX_PATIENCE = args.max_patience
    cfg.VERBOSE = True

    # Test
    # cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # INPUT
    image_size = args.image_size  # 1024
    cfg.INPUT.MIN_SIZE_TRAIN = (image_size,)
    cfg.INPUT.MIN_SIZE_TEST = image_size

    # DATASETS
    # cfg.DATASETS.TEST = ('val',)
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATALOADER.NUM_WORKERS = 8

    # SAVE MODEL
    type_path = (
        "transferred_learning"
        if args.load_image_caption
        else "non_transferred_learning"
    )
    val_loss_filter = "_".join(args.validation_loss)
    cfg.OUTPUT_DIR = f"{args.save_dir}/{type_path}/{args.dataset_name}_Dataset/{args.model}/{image_size}_{val_loss_filter}_loss"
    cfg.BEST_MODEL_DIR = cfg.OUTPUT_DIR + "/early_stopped_model.pth"
    cfg.SAVE_MODEL = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        args.threshold  # set the testing threshold for this model
    )

    # PERIODIC HOOK
    args.periodic_hook_add = os.path.join(
        args.periodic_hook_add
        + f"/{args.dataset_name}_Dataset/{args.model}/{image_size}_{val_loss_filter}_loss"
    )
    if not os.path.exists(args.periodic_hook_add):
        os.makedirs(args.periodic_hook_add)

    # INFERENCE
    args.infer_dir = os.path.join(
        args.infer_dir
        + f"/{args.dataset_name}_Dataset/{args.model}/{image_size}_{val_loss_filter}_loss/Inference"
    )
    if not os.path.exists(args.infer_dir):
        os.makedirs(args.infer_dir)

    return args, cfg


class transformers_config:
    def __init__(self):
        # Backbone
        self.args, self.backbone_cfg = detectron_config(
            ctgry_pre_address="segmentation/"
        )
        self.backbone_name = self.backbone_cfg["MODEL"]["WEIGHTS"].split("/")[-3]
        self.args.data_path = "/media/amir_shirian/abd1fa2e-8fdf-46e4-9dbf-dc0a070ba9b6/home/user/Desktop/Segmentation"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.RESIZE = 512  # 356
        self.CROP = 512  # 299
        self.early_stopping_patience = 6

        # Model Hyperparameters
        self.batch_size = 4
        self.max_length_caption = 470
        self.dropout = 0.5
        self.learning_rate = 1e-03
        self.load_model = True
        self.save_model = True
        self.train_CNN = False
        self.alpha_c = 1
        self.step = 0
        # for tensorboard
        # writer = SummaryWriter("runs/flickr")

        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 1500
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Transformer Backbone
        self.backbone = "resnet101"
        self.position_embedding = "sine"
        self.dilation = True

        # Basic
        self.device = "cuda"
        # self.seed = 42
        self.num_workers = 8
        self.save_dir = "./trained_models/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.checkpoint = (
            self.save_dir
            + f"transformer_model_with_{self.backbone_name}_checkpoint.pth"
        )
        self.backbone_checkpoint = (
            self.save_dir + f"backbone_model_{self.backbone_name}_checkpoint.pth"
        )
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        # self.max_position_embeddings = 128
        self.max_position_embeddings = self.max_length_caption
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522
        self.config = None

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        self.limit = -1
