import os
from random import random

import cv2
import matplotlib.pyplot as plt
import wandb
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer

from configuration import detectron_config, init_wab
from segmentation.data.dataloader import load_data
from segmentation.utils.hooks import PeriodicImgSegHook, ValidationHook, WabLogHook

if __name__ == "__main__":
    args, cfg = detectron_config(ctgry_pre_address="")

    # initialize wandb
    if args.wab_project is None:
        val_loss_filter = "_".join(args.validation_loss)
        args.wab_project = f"Image_Segmentation_{args.dataset_name}_Dataset.{args.model}.{args.image_size}_Img_size.{val_loss_filter}_loss"
    if args.enable_wab:
        init_wab(
            wab_config_path=args.wab_config_path,
            model_config=cfg,
            model_args=args,
            entity=args.wab_entity,
            project_name=args.wab_project,
            key=args.wab_key,
        )

    for x in ["train", "val"]:
        DatasetCatalog.register(
            x,
            lambda x=x: load_data(
                cfg, portion=x, pre_address=args.data_path, dataset_name=args.dataset_name
            ),
        )
        MetadataCatalog.get(x).set(thing_classes=cfg.CATEGORIES)
    metadata = MetadataCatalog.get("train")

    transferred_learning = "non_transferred_learning"
    if args.load_image_caption:
        cfg.MODEL.WEIGHTS = args.image_caption_model_add
        transferred_learning = "transferred_learning"

    trainer = DefaultTrainer(cfg)
    cfg.DATASETS.VAL = ("val",)

    # create output directory if it does not exist
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # add image segmentation hook
    img_seg_hook = PeriodicImgSegHook(
        cfg,
        eval_period=args.periodic_hook_period,
        path=args.periodic_hook_add,
        metadata=metadata,
        wandb_log=args.enable_wab,
    )
    trainer.register_hooks([img_seg_hook])

    # add validation hook
    val_hook = ValidationHook(
        cfg,
        eval_period=2500 // (cfg.SOLVER.IMS_PER_BATCH * 2),
        filter=args.validation_loss,
    )
    trainer.register_hooks([val_hook])

    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # add wandb logging hook
    if args.enable_wab:
        trainer.register_hooks([WabLogHook(cfg, eval_period=100)])

    # train
    trainer.resume_or_load(resume=False)
    trainer.train()

    # load the best model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.BEST_MODEL_DIR)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        args.threshold  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("val",)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = load_data(
        cfg, portion="val", pre_address=args.data_path, dataset_name=args.dataset_name
    )

    for d in dataset_dicts[:20]:
        fig, axs = plt.subplots(1, 2)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )  # remove the colors of unsegmented pixels)
        v_gt = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )  # remove the colors of unsegmented pixels)
        gt = v_gt.draw_dataset_dict(d)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        axs[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[1].imshow(cv2.cvtColor(gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[0].set_title("Prediction")
        axs[1].set_title("Ground Truth")
        # save fig
        fig_name = d["file_name"].split(".")[-2].split("/")[-1]
        plt.savefig(
            f"trained_models/{transferred_learning}/CANDIDPTX_models/{cfg.args.model}/{fig_name}.png",
            dpi=900,
        )
        plt.close()

    # Call wandb.finish() to upload your TensorBoard logs to W&B
    wandb.finish()
