import copy

import albumentations as A
import numpy as np
import torch
from detectron2.data import detection_utils as utils


class AlbumentationsMapper:
    # Mapper which uses `albumentations` augmentations
    def __init__(self, cfg, is_train: bool = True):
        # aug_kwargs = cfg.aug_kwargs
        aug_kwargs = {
            "HorizontalFlip": {"p": 0.5},
            "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
            "RandomBrightnessContrast": {"p": 0.5},
            # "ToPilImage": {},
            # "Resize": {"height": 356, "width": 356, "always_apply": True},
            # "RandomCrop": {"height": 299, "width": 299, "always_apply": True},
            # "HorizontalFlip": {"p": 0.5},
            # "ColorJitter": {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0, "always_apply": False},
            # "ToTensor": {"always_apply": True},
            # "Normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "max_pixel_value": 255.0, "always_apply": True},
        }
        aug_list = []
        if is_train:
            aug_list.extend(
                [getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()]
            )
        self.transform = A.Compose(
            aug_list,
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
        )

        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(
            dataset_dict
        )  # it will be modified by the code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        prev_anno = dataset_dict["annotations"]
        bboxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)
        category_id = np.arange(len(dataset_dict["annotations"]))
        # for i in range(len(bboxes)):
        #     for j in range(4):
        #         if bboxes[i][j] < 0:
        #             bboxes[i][j] = 0
        #         # elif bboxes[i][j] > 1:
        #         #     bboxes[i][j] = 1
        transformed = self.transform(
            image=image, bboxes=bboxes, category_ids=category_id
        )
        image = transformed["image"]
        annos = []
        for i, j in enumerate(transformed["category_ids"]):
            d = prev_anno[j]
            d["bbox"] = transformed["bboxes"][i]
            annos.append(d)
        dataset_dict.pop("annotations", None)  # Remove unnecessary field.

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
