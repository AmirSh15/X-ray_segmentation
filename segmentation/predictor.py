import os

import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

test_data = [
    {"file_name": ".../image_1.jpg", "image_id": 10},
    {"file_name": ".../image_2.jpg", "image_id": 20},
]


def visualization(metadata, cfg, test_set):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    for d in test_set:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(
            os.path.join(
                os.path.join(cfg.OUTPUT_DIR, "visualization"),
                str(d["image_id"]) + ".png",
            ),
            img,
        )
