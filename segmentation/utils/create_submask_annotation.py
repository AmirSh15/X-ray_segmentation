import numpy as np  # (pip install numpy)
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
from shapely.geometry import MultiPolygon, Polygon  # (pip install Shapely)
from skimage import measure  # (pip install scikit-image)


def create_sub_mask_annotation(
    polygons,
    boxes,
    syms,
    image_id,
    categoris,
    coco,
    coco_image,
    annot_list,
    annotation_id,
):

    category_id = 0
    annotations = []
    if polygons:
        for c, contour in enumerate(polygons):
            contours = []
            segmentations = []
            for cat in categoris:
                if cat["name"] == syms[c]:
                    category_id = cat["id"]
                    break

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            # check for difference between real bbox and generated one

            if type(poly).__name__ == "MultiPolygon":
                segmentation = []
                for p in poly:
                    contours.append(p)
                    segmentation.append(np.array(p.exterior.coords).ravel().tolist())
                    segmentations = segmentation
            else:
                contours.append(poly)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)

            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(contours)
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area

            annot_list.append(
                {
                    "segmentation": segmentations,
                    # 'iscrowd': is_crowd,
                    "iscrowd": 0,
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "id": len(annot_list),
                    # 'bbox': boxes[c],
                    "bbox": bbox,
                    "area": area,
                }
            )
            annotation_id += 1
    else:
        annotations.append([])
    coco.add_image(coco_image)
    return annotations, annotation_id
