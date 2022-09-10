import os
from random import random

import cv2
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.utils.visualizer import Visualizer

from configuration import detectron_config
# from segmentation.data.dataloader import load_data
from segmentation.data.dataloader_CANDIDPTX import load_data
from segmentation.utils.augmentations import AlbumentationsMapper
from segmentation.utils.validation_loss import ValidationLoss


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=AlbumentationsMapper(cfg, True), sampler=sampler
        )


if __name__ == "__main__":

    cfg = detectron_config(ctgry_pre_address="")

    for d in ["train", "val"]:
        DatasetCatalog.register(d, lambda d=d: load_data(d, ""))
        # MetadataCatalog.get(d).set(thing_classes=cfg.CATEGORIES)
        MetadataCatalog.get(d).set(thing_classes=["phnx"])
    metadata = MetadataCatalog.get("train")

    # cfg.MODEL.WEIGHTS = "../captioning_transformer/trained_models/backbone_model_mask_rcnn_R_101_C4_3x_checkpoint.pth"
    # trainer = DefaultTrainer(cfg)
    trainer = CustomTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    cfg.DATASETS.VAL = ("val",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    val_loss = ValidationLoss(cfg, eval_period=2500 // (cfg.SOLVER.IMS_PER_BATCH * 2))
    trainer.register_hooks([val_loss])

    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    # trainer.train()
    # https: // github.com / Sagar - py / chestXRay - Detection / blob / main / notebook - train.ipynb

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###########################
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    #
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.WEIGHTS = 'model_final.pth'
    #
    # evaluator = COCOEvaluator("validation_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # val_loader = build_detection_test_loader(cfg, "validation_dataset")
    # valResults = inference_on_dataset(trainer.model, val_loader, evaluator)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~############################

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.BEST_MODEL_DIR)
    # cfg.MODEL.WEIGHTS = "trained_models/transferred/CANDIDPTX_models/model_final.pth"
    cfg.MODEL.WEIGHTS = (
        "trained_models/transferred/CANDIDPTX_models/best_model/early_stopped_model.pth"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("val",)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = load_data("val")
    # dataset_dicts = load_data("train")
    # for d in random.sample(dataset_dicts, 50):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.figure(figsize=(14, 10))
    #     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    #     plt.show()

    from detectron2.utils.visualizer import ColorMode

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
        # plt.figure(figsize=(14, 10))
        axs[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[1].imshow(cv2.cvtColor(gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[0].set_title("Prediction")
        axs[1].set_title("Ground Truth")
        # save fig
        fig_name = d["file_name"].split(".")[-2].split("/")[-1]
        plt.savefig(
            f"trained_models/transferred/CANDIDPTX_models/best_model/{fig_name}.png",
            dpi=900,
        )
        plt.close()
