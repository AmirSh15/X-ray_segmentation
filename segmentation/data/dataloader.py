import json
import os

import detectron2.data.datasets.coco as dscoco
import numpy as np
import pandas as pd
import PIL
import pydicom as dicom
from PIL import Image
from pycocotools import mask
from sahi.utils.coco import Coco, CocoImage
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

# from create_submask_annotation import create_sub_mask_annotation as csma
from configuration import detectron_config
from segmentation.utils.create_submask_annotation import \
    create_sub_mask_annotation as csma


def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(" ")])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start:end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def mask2poly(mask_list, image_id, annotation_id):
    fortran_ground_truth_binary_mask = np.asfortranarray(mask_list)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(mask_list, 0.5)

    # ~~~~~~~~~~~~~~~~##############################
    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = [x, y, width, height]
    area = multi_poly.area

    annot = {
        "segmentation": segmentations,
        "iscrowd": 0,
        "image_id": image_id,
        "category_id": 0,
        "id": annotation_id,
        "bbox": bbox,
        "area": area,
    }

    return annot


def create_coco_dataset_CANDIDPTX(cfg, data, image_path, ds_type, pre_address):

    # specify the categories
    categories = cfg.CATEGORIES

    # second: 'annotations' and 'images' data
    record = {}
    coco = Coco()
    annotations = []
    images = []
    annotation_id = 0
    start = 0
    end = 0
    if ds_type == "train":
        start = 0
        end = int(len(os.listdir(image_path)) * 0.8)
    if ds_type == "val":
        start = int(len(os.listdir(image_path)) * 0.8)
        end = len(os.listdir(image_path))

    for img_idx, img in enumerate(os.listdir(image_path)[start:end], start=start):
        img_name = str(img).strip(".png")

        rle_list = data[data["SOPInstanceUID"] == img_name]["EncodedPixels"]
        record["file_name"] = str(img)
        record["id"] = int(img_idx)
        # im = PIL.Image.open(image_path + record['file_name'])
        # record['width'], record['height'] = im.size
        record["width"], record["height"] = 1024, 1024
        coco_image = CocoImage(
            file_name=record["file_name"],
            height=record["height"],
            width=record["width"],
        )

        for annot_idx, rle in enumerate(rle_list):
            neg_flag = False
            x = len(rle_list)
            if rle != "-1" and isinstance(rle, str):
                component = run_length_decode(
                    rle, height=record["width"], width=record["height"]
                ).astype(np.uint8)
                converted_rle = mask2poly(component, img_idx, annotation_id)

                for i in range(len(converted_rle["bbox"])):
                    if converted_rle["bbox"][i] < 0:
                        # neg_flag = True
                        converted_rle["bbox"][i] = 0
                        break

                for j in range(len(converted_rle["segmentation"])):
                    for i in range(len(converted_rle["segmentation"][j])):
                        if converted_rle["segmentation"][j][i] < 0:
                            # neg_flag = True
                            break
                            # converted_rle['segmentation'][j][i] = 0

                if not neg_flag:
                    annotations.append(converted_rle)
                    annotation_id += 1
                else:
                    continue

            else:
                continue

        images.append(record)
        coco.add_image(coco_image)
        record = {}

    # third: 'info' data
    info = dict(
        url="https://pubs.rsna.org/doi/10.1148/ryai.2021210136",
        contributor="Not valid",
        year=2021,
        description="alpha",
        date_created="2021",
        version=1.0,
    )

    # forth: 'licenses' data
    licenses = ["private"]

    # put data in final dictionary
    data = dict(
        categories=categories,
        annotations=annotations,
        info=info,
        images=images,
        licenses=licenses,
    )
    with open(
        pre_address
        + "/CANDID_PTX/annotations/new_CANDIDPTX_"
        + ds_type
        + "_coco_style_annot.json",
        "w",
    ) as outfile:
        json.dump(data, outfile)

    return data


def create_coco_dataset(cfg, data, image_path, ds_type, pre_address):

    # first: 'categories' data
    all_cats = cfg.CATEGORIES

    categories = []
    for i, cat in enumerate(all_cats):
        category = {}
        category["supercategory"] = None
        category["id"] = i
        category["name"] = cat
        categories.append(category)

    # second: 'annotations' and 'images' data
    record = {}
    coco = Coco()
    annotations = []
    images = []
    annotation_id = 0
    for i in range(len(data)):
        # record['file_name'] = image_path+data[i]['file_name']
        record["file_name"] = data[i]["file_name"]

        record["id"] = int(data[i]["file_name"].strip(".png"))
        image = PIL.Image.open(image_path + record["file_name"])
        width, height = image.size
        record["width"] = width
        record["height"] = height
        coco_image = CocoImage(
            file_name=record["file_name"], height=height, width=width
        )

        csma(
            data[i]["polygons"],
            data[i]["boxes"],
            data[i]["syms"],
            int(data[i]["file_name"].strip(".png")),
            categories,
            coco,
            coco_image,
            annotations,
            annotation_id,
        )
        # dataset.append(record)
        images.append(record)
        record = {}

    # third: 'info' data
    info = dict(
        url="https://github.com/Deepwise-AILab/ChestX-Det-Dataset",
        contributor="Not valid",
        year=2021,
        description="alpha",
        date_created="2021",
        version=1.0,
    )

    # forth: 'licenses' data
    licenses = ["private"]

    # put data in final dictionary
    data = dict(
        categories=categories,
        annotations=annotations,
        info=info,
        images=images,
        licenses=licenses,
    )

    with open(
        pre_address
        + "data/ChestX_Det/annotations/"
        + ds_type
        + "_coco_style_annot.json",
        "w",
    ) as outfile:
        json.dump(data, outfile)

    return data


def load_data(cfg, portion="train", pre_address="", dataset_name="CANDIDPTX"):

    if dataset_name == "ChestX-Det":
        if portion == "train":
            with open(
                pre_address + "data/ChestX_Det/annotations/ChestX_Det_train.json", "r"
            ) as file:
                data = json.load(file)
                data = [d for d in data if len(d["boxes"]) != 0]
                train_images_path = pre_address + "data/ChestX_Det/images/train/"
                create_coco_dataset(cfg, data, train_images_path, portion, pre_address)

                # train_dict = dscoco.load_coco_json('data/annotations/train_coco_style_annot.json', train_images_path)
                train_dict = dscoco.load_coco_json(
                    pre_address
                    + "data/ChestX_Det/annotations/train_coco_style_annot.json",
                    train_images_path,
                )

                return train_dict

        elif portion == "val":
            with open(
                pre_address + "data/ChestX_Det/annotations/ChestX_Det_test.json", "r"
            ) as file:
                data = json.load(file)
                data = [d for d in data if len(d["boxes"]) != 0]
                val_images_path = pre_address + "data/ChestX_Det/images/test/"
                create_coco_dataset(cfg, data, val_images_path, portion, pre_address)

                train_dict = dscoco.load_coco_json(
                    pre_address
                    + "data/ChestX_Det/annotations/val_coco_style_annot.json",
                    val_images_path,
                )
                return train_dict
    elif dataset_name == "CANDIDPTX":
        data = pd.read_csv(
            pre_address + "/CANDID_PTX/Pneumothorax_reports.csv"
        )

        if portion == "train":
            # with open(pre_address + "data/annotations/train_coco_style_annot_CANDIDPTX", 'r') as file:
            #     data = json.load(file)
            train_images_path = pre_address + "/CANDID_PTX/images/"
            if not os.path.exists(pre_address
                + "/CANDID_PTX/annotations/new_CANDIDPTX_train_coco_style_annot.json"):
                create_coco_dataset_CANDIDPTX(
                    cfg, data, train_images_path, portion, pre_address
                )

            train_dict = dscoco.load_coco_json(
                pre_address
                + "/CANDID_PTX/annotations/new_CANDIDPTX_train_coco_style_annot.json",
                train_images_path,
            )

            return train_dict

        elif portion == "val":
            val_images_path = pre_address + "/CANDID_PTX/images/"
            if not os.path.exists(pre_address
                + "/CANDID_PTX/annotations/new_CANDIDPTX_val_coco_style_annot.json"):
                create_coco_dataset_CANDIDPTX(
                    cfg, data, val_images_path, portion, pre_address
                )

            val_dict = dscoco.load_coco_json(
                pre_address
                + "/CANDID_PTX/annotations/new_CANDIDPTX_val_coco_style_annot.json",
                val_images_path,
            )
            return val_dict
