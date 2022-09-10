import logging
import os

import cv2
import detectron2.utils.comm as comm
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.data import (build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import HookBase
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm
import wandb
from PIL import Image, ImageDraw


class PeriodicImgSegHook(HookBase):
    def __init__(self, cfg, eval_period, path, metadata, num_images=10, wandb_log=False,):
        super().__init__()
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._dataloader = build_detection_train_loader(self.cfg)
        self._loader = self._dataloader.dataset.dataset.dataset
        self._dataset = self._loader._dataset
        self.path = path
        self.metadata = metadata
        self.class_names_wo_backgrd = {id + 1: cat for id, cat in enumerate(cfg.CATEGORIES)}
        self.class_names = self.class_names_wo_backgrd.copy()
        self.class_names.update({0: "background"})
        self.verbose = cfg.VERBOSE
        self.num_images = num_images
        self.wandb_log = wandb_log

    def _do_save_images(self):
        self.trainer.model.eval()
        dir_path = os.path.join(self.path, "iter_{}".format(self.trainer.iter+1))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with torch.no_grad():
            mask_img_list = []
            for idx, (data, org_data) in enumerate(
                tqdm(
                    zip(self._loader, self._dataset),
                    total=self.num_images,
                    leave=False,
                    desc="Saving images",
                )
            ):
                predictions = self.trainer.model([data])[0]
                im = cv2.imread(data["file_name"])
                fig, axs = plt.subplots(1, 2)
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=self.metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE_BW,
                )  # remove the colors of unsegmented pixels)
                v_gt = Visualizer(
                    im[:, :, ::-1],
                    metadata=self.metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE_BW,
                )  # remove the colors of unsegmented pixels)
                gt = v_gt.draw_dataset_dict(org_data)
                v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
                axs[0].imshow(
                    cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
                )
                axs[1].imshow(
                    cv2.cvtColor(gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
                )
                axs[0].set_title("Prediction")
                axs[1].set_title("Ground Truth")
                fig.tight_layout()

                # save fig
                fig_name = data["file_name"].split("/")[-1]
                plt.savefig(
                    dir_path + f"/{idx}_{fig_name}",
                    dpi=900,
                )
                plt.close()

                # log to wandb
                if self.wandb_log:
                    # extract the predicted mask
                    if len(predictions["instances"].to("cpu")) == 0:
                        mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
                    else:
                        mask = np.zeros((im.shape[0], im.shape[1]))
                        for poly, class_id in zip(predictions["instances"].to("cpu").get_fields()['pred_masks'], predictions["instances"].to("cpu").get_fields()['pred_classes']):
                            polygon = np.array(poly, dtype=np.int)
                            mask += polygon * (class_id.item() + 1)

                    # extract the ground truth mask
                    gt_mask = Image.new("L", (im.shape[0], im.shape[1]), 0)
                    for poly in org_data['annotations']:
                        polygon = np.array(poly['segmentation'][0], dtype=np.int).flatten().tolist()
                        # polygon = np.array(poly, dtype=np.int).flatten().tolist()
                        polygon = [
                            (polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)
                        ]
                        ImageDraw.Draw(gt_mask).polygon(polygon, outline=1, fill=poly['category_id'] + 1)
                    gt_mask = np.array(gt_mask)

                    mask_img = wandb.Image(im, masks={
                        "prediction": {
                            "mask_data": mask,
                            "class_labels": self.class_names_wo_backgrd,
                        },
                        "ground_truth": {
                            "mask_data": gt_mask,
                            "class_labels": self.class_names,
                        },
                    })
                    mask_img_list.append(mask_img)

                    wandb.log({f"iter_{self.trainer.iter+1}/"+data["file_name"].split('/')[-1]: mask_img}, step=self.trainer.iter, commit=False)

                if idx == self.num_images - 1:
                    self.trainer.model.train()
                    # if self.wandb_log:
                    #     wandb.log({"predictions": mask_img_list}, step=self.trainer.iter, commit=False)
                    break

    def after_step(self):
        next_iter = int(self.trainer.iter) + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval_period > 0 and next_iter % self.eval_period == 0):
            self._do_save_images()
        else:
            pass


class ValidationHook(HookBase):
    def __init__(self, cfg, eval_period, filter=["val_"]):
        super().__init__()
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._dataloader = build_detection_train_loader(self.cfg)
        self._dataset = self._dataloader.dataset.dataset.dataset
        self._loader = iter(self._dataloader)
        self._best_score = float("inf")
        self.val_loss_min = np.Inf
        self._max_patience = cfg.MAX_PATIENCE
        self.path = cfg.BEST_MODEL_DIR
        self._patience = 0
        self.verbose = cfg.VERBOSE
        self.filter = filter

    def _do_loss_eval(self):
        loss_dict_sum = {}
        count = 0
        with torch.no_grad():
            for idx, data in enumerate(
                tqdm(self._dataset, total=len(self._dataset), leave=False, desc="Evaluating")
            ):
                if idx == 0:
                    loss_dict_sum = self.trainer.model([data])
                else:
                    loss_dict = self.trainer.model([data])
                    loss_dict_sum = {
                        k: loss_dict_sum[k] + loss_dict[k] for k in loss_dict_sum.keys()
                    }
                count += 1

            losses = sum(loss_dict_sum.values()) / count
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item() / count
                for k, v in comm.reduce_dict(loss_dict_sum).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )

            losses_reduced = sum(
                loss
                for name, loss in loss_dict_reduced.items()
                if all(x in name for x in self.filter)
            )

            if self.val_loss_min > losses_reduced:
                logger = logging.getLogger(__name__)
                if self.verbose:
                    logger.info(
                        f"Validation Loss Decreased({self.val_loss_min:.6f}--->{losses_reduced:.6f}) \t Saving The Model"
                    )

                self._patience = 0
                self.val_loss_min = losses_reduced
                if self.cfg.SAVE_MODEL:
                    # if True:
                    torch.save(self.trainer.model.state_dict(), self.cfg.BEST_MODEL_DIR)

            else:
                self._patience += 1
                logger = logging.getLogger(__name__)
                if self.verbose:
                    logger.info(
                        f"Early stopping patience: {self._patience} out of {self._max_patience}"
                    )
                if self._patience >= self._max_patience:
                    # self.trainer.max_iter = self.trainer.iter
                    # self.trainer.iter = self.trainer.max_iter
                    raise StopIteration

    def after_step(self):
        next_iter = int(self.trainer.iter) + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval_period > 0 and next_iter % self.eval_period == 0):
            mean_loss = self._do_loss_eval()
        else:
            pass

class WabLogHook(HookBase):
    def __init__(self, cfg, eval_period):
        super().__init__()
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.verbose = cfg.VERBOSE

    def _do_log(self):
        # log to wandb
        wandb_filter = ["loss", "lr"]
        log_dict = {k: v[0] for k, v in self.trainer.storage.latest().items() if
                    any([f in k for f in wandb_filter])}
        wandb.log(log_dict, step=self.trainer.iter + 1, commit=False)

    def after_step(self):
        next_iter = int(self.trainer.iter) + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval_period > 0 and next_iter % self.eval_period == 0):
            self._do_log()
        else:
            pass