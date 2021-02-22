from typing import List, Optional
from detectron2.utils.visualizer import ColorMode
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T


def visualize_det2(dataset_dicts: List[dict],
                   predictor: DefaultPredictor,
                   metadata: Optional[Metadata] = None,
                   count: int = 10) -> None:
    for d in random.sample(dataset_dicts, count):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            scale=0.5,
            metadata=metadata,
            instance_mode=ColorMode.
            IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def create_train_augmentation(cfg):
    augs = [
        T.Resize((cfg.image_h, cfg.image_w)),
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.3)
        # T.RandomCrop("absolute", (640, 640))
    ]
    return augs


def create_test_augmentation(cfg):
    augs = [
        T.Resize((cfg.image_h, cfg.image_w)),
    ]
    return augs