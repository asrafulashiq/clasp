from typing import Dict, List, Mapping
from detectron2.structures import BoxMode
import os
from pathlib import Path
import numpy as np
import parse
import json
from tqdm import tqdm
import cv2
import random


def extract_file_info(filename: str) -> Mapping:
    # filename: cam2_vid2_seg7.json
    parsed_info = parse.parse("cam{cam}_vid{vid}_seg{seg}.json", str(filename))
    return parsed_info


def scale_kp_xy(kp: List[float], w, h) -> List[float]:
    n = len(kp) // 2
    new_kp = [1] * (3 * n)
    new_kp[0::3] = [(i * w) if i else i for i in kp[0::2]]
    new_kp[1::3] = [(i * h) if i else i for i in kp[1::2]]
    new_kp[2::3] = [1 if i is not None else 0 for i in kp[0::2]]
    return new_kp


def get_conflab_dict(img_root_dir: str, annotation_dir: str) -> List[Dict]:
    counter_image = 0
    for ann_file in os.listdir(annotation_dir)[5:]:
        parsed_info = extract_file_info(ann_file)

        img_dir = os.path.join(
            img_root_dir,
            f"cam{parsed_info['cam']}/vid{parsed_info['vid']}-seg{parsed_info['seg']}-scaled-denoised"
        )
        assert os.path.exists(img_dir)

        with open(os.path.join(annotation_dir, ann_file), 'r') as fp:
            data = json.load(fp)

        dataset_dicts = []

        for idx, v in tqdm(enumerate(data)):
            # v contain info for each image
            ann_image = list(v.items())
            record = {}
            filename = os.path.join(
                img_dir, f"{ann_image[0][1]['image_id']+1:06d}.jpg")
            assert os.path.exists(filename)

            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = counter_image
            record["height"] = height
            record["width"] = width

            objs = []
            for _, anno in ann_image:
                anno["keypoints"] = scale_kp_xy(anno["keypoints"], width,
                                                height)
                px = anno["keypoints"][0::3]
                py = anno["keypoints"][1::3]
                poly = [(x + 0.5, y + 0.5) if x and y else (x, y)
                        for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                fn_none = lambda x: [i for i in x if i is not None]
                obj = {
                    "bbox": [
                        np.min(fn_none(px)),
                        np.min(fn_none(py)),
                        np.max(fn_none(px)),
                        np.max(fn_none(py))
                    ],
                    "bbox_mode":
                    BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id":
                    0,
                    "keypoints":
                    anno["keypoints"],
                    "iscrowd":
                    0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

            counter_image += 1
            if counter_image > 100:
                return dataset_dicts
        break
    return dataset_dicts


# --------------------------------- Register --------------------------------- #
from detectron2.data import DatasetCatalog

DATASET_NAME = "conflab-dataset"
IMG_ROOT_DIR = "/home/ash/datasets/conflab-mm/frames/videoSegments"
ANN_DIR = "/home/ash/datasets/conflab-mm/annotations"

DatasetCatalog.register(
    DATASET_NAME, lambda: get_conflab_dict(img_root_dir=IMG_ROOT_DIR,
                                           annotation_dir=ANN_DIR))

if __name__ == "__main__":
    from detectron2.utils.visualizer import Visualizer

    dataset_dicts = get_conflab_dict(
        img_root_dir="/home/ash/datasets/conflab-mm/frames/videoSegments",
        annotation_dir="/home/ash/datasets/conflab-mm/annotations")

    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_im = out.get_image()[:, :, ::-1]

        cv2.imshow(d["file_name"], cv2_im)
        cv2.waitKey(0)

    cv2.destroyAllWindows()