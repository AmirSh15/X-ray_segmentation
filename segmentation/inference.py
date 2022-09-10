import os

import cv2
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import (COCOEvaluator, LVISEvaluator,
                                   inference_on_dataset)
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt
from utils.evaluator import CustomSemSegEvaluator

from configuration import detectron_config
from segmentation.data.dataloader import load_data

if __name__ == "__main__":
    args, cfg = detectron_config(ctgry_pre_address="")

    for x in ["train", "val"]:
        DatasetCatalog.register(
            x,
            lambda x=x: load_data(
                cfg, portion=x, pre_address=args.data_path, dataset_name=args.dataset_name
            ),
        )
        MetadataCatalog.get(x).set(thing_classes=cfg.CATEGORIES)
    metadata = MetadataCatalog.get("val")

    cfg.MODEL.WEIGHTS = os.path.join(args.infer_model_add)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        args.threshold  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("val",)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = load_data(
        cfg, portion="val", pre_address=args.data_path, dataset_name=args.dataset_name
    )
    cfg.DATASETS.TRAIN = ("val",)
    # val_loader = build_detection_train_loader(cfg)
    val_loader = build_detection_test_loader(cfg, "val")

    eval_path = os.path.join(cfg.OUTPUT_DIR, "eval")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path, exist_ok=True)
    AP_evaluator = COCOEvaluator("val", cfg, False, output_dir=eval_path)
    # LVIS_evaluator = LVISEvaluator("val")
    SemSeg_evaluator = CustomSemSegEvaluator("val", False, output_dir=eval_path)
    # result = inference_on_dataset(
    #     predictor.model,
    #     val_loader,
    #     [
    #         AP_evaluator,
    #         # LVIS_evaluator,
    #         SemSeg_evaluator,
    #     ],
    # )

    for idx, data in enumerate(dataset_dicts[:20]):
        fig, axs = plt.subplots(1, 2)
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,
        )  # remove the colors of unsegmented pixels)
        v_gt = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,
        )  # remove the colors of unsegmented pixels)
        gt = v_gt.draw_dataset_dict(data)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        axs[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[1].imshow(cv2.cvtColor(gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        axs[0].set_title("Prediction")
        axs[1].set_title("Ground Truth")
        fig.tight_layout()

        # save fig
        fig_name = data["file_name"].split("/")[-1]
        plt.savefig(
            args.infer_dir + f"/{idx}_{fig_name}",
            dpi=900,
        )
        plt.close()
