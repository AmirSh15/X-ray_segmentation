import logging

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import SemSegEvaluator
from PIL import Image, ImageDraw


class CustomSemSegEvaluator(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        num_classes=None,
        ignore_label=None,
    ):
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_semseg_gt = {
            dataset_record["file_name"]: [
                x["segmentation"][0] for x in dataset_record["annotations"]
            ]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = [c for c in meta.thing_classes]
        self._num_classes = len(self._class_names)
        if num_classes is not None:
            assert (
                self._num_classes == num_classes
            ), f"{self._num_classes} != {num_classes}"
        # self._ignore_label = (
        #     ignore_label if ignore_label is not None else meta.ignore_label
        # )
        self._ignore_label = ignore_label

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            img = Image.new("L", (input["width"], input["height"]), 0)

            if len(self.input_semseg_gt[input["file_name"]]) > 1:
                for poly in self.input_semseg_gt[input["file_name"]]:
                    polygon = np.array(poly, dtype=np.int)
                    polygon = [
                        (polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)
                    ]
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            else:
                polygon = self.input_semseg_gt[input["file_name"]][0]
                polygon = np.array(polygon, dtype=np.int)
                polygon = [
                    (polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)
                ]
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

            if output["instances"].get_fields()["pred_masks"].shape[0] == 0:
                output = np.zeros((input["height"], input["width"]), dtype=np.uint8)
            else:
                output = (
                    output["instances"].get_fields()["pred_masks"].to(self._cpu_device)
                )

            pred = np.array(output, dtype=np.int)
            gt = np.array(img)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
