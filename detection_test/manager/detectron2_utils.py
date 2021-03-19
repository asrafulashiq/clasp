import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from torchvision.transforms import functional as F
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
import cv2
from tools.utils import BinType

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_RED = (255, 10, 10)
_BLUE = (10, 10, 255)

class_to_color = {"item": _GREEN, "bin_empty": _RED}
# clas_idx_to_str = {0: "bin_empty", 1: "item"}


def format_boxes(boxes, scores, classes, thresh=0.8):
    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    _boxes = []
    _score = []
    _class = []

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores[i]
        # clss = clas_idx_to_str[int(classes[i])]
        try:
            clss = BinType(int(classes[i]))
        except ValueError as e:
            raise ValueError(f"{classes[i]} not a valid type") from e

        if score < thresh:
            continue
        _boxes.append(bbox)
        _score.append(score)
        _class.append(clss)

    return np.array(_boxes), np.array(_score), np.array(_class, dtype=BinType)


class DetectorObj:
    def __init__(self, args, labels_to_keep=(0, 1)):
        self.args = args

        self.create_model()

    def create_model(self):
        self.model = RCNN_Detector(self.args, thresh=self.args.thresh)

    def predict_box(self, im: np.ndarray, show=False) -> Tuple:
        ret = self.model(im)
        if ret is not None:
            boxes, scores, classes = ret
            nim = im
            boxes, scores, classes = format_boxes(boxes,
                                                  scores,
                                                  classes,
                                                  thresh=self.args.thresh)
            return nim, boxes, scores, classes
        else:
            return im, None, None, None

    def predict_box_batch(self,
                          imlist: List[np.ndarray],
                          show: bool = False,
                          max_batch: int = 10) -> List[Tuple]:
        rets = []
        strt = 0
        pbar = tqdm(total=len(imlist), desc="DET ", leave=False)
        while True:
            _end = min(strt + max_batch, len(imlist))
            ims = [imlist[i] for i in range(strt, _end)]
            _ret = self.model.predict_batch(ims)
            rets.extend(_ret)
            pbar.update(len(ims))
            if _end >= len(imlist):
                break
            strt = _end
        pbar.clear()
        pbar.close()

        results = []
        for i, ret in enumerate(rets):
            im = imlist[i]
            if ret is not None:
                boxes, scores, classes = ret
                nim = imlist[i]
                # if show:
                boxes, scores, classes = format_boxes(boxes,
                                                      scores,
                                                      classes,
                                                      thresh=self.args.thresh)
                results.append((im, boxes, scores, classes))
            else:
                results.append((im, None, None, None))
        return results


class RCNN_Detector():
    def __init__(self, args, labels_to_keep=(0, 1), thresh=0.8):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')

        self.args = args
        self.cfg = self.setup(args)

        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.model.to(self.device)

        self.labels_to_keep = labels_to_keep
        self.thresh = thresh

    def setup(self, args):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = args.bin_ckpt
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh  # set a custom testing threshold
        return cfg

    @torch.no_grad()
    def predict_one(self, im):
        height, width = im.shape[:2]
        imt = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs = [{
            "image": imt.to(self.device),
            "height": height,
            "width": width
        }]
        output = self.model(inputs)
        # return output[0]

        output = {
            k: v
            for k, v in self.get_ret_from_prediction(output[0]['instances'].to(
                "cpu")).items()
        }
        index = (np.isin(output["pred_classes"], self.labels_to_keep) &
                 (output["scores"] > self.thresh))
        if index.size > 0:
            boxes = output["pred_boxes"][index]
            scores = output["scores"][index]
            classes = output["pred_classes"][index]
            return boxes, scores, classes
        else:
            return None

    @torch.no_grad()
    def predict_batch(self, imlist):
        inputs = []
        for im in imlist:
            height, width = im.shape[:2]
            imt = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
            inputs.append({
                "image": imt.to(self.device),
                "height": height,
                "width": width
            })

        batch_output = self.model(inputs)
        results = []
        for i in range(len(imlist)):
            output = {
                k: v
                for k, v in self.get_ret_from_prediction(
                    batch_output[i]['instances'].to("cpu")).items()
            }
            index = (np.isin(output["pred_classes"], self.labels_to_keep) &
                     (output["scores"] > self.thresh))
            if index.size > 0:
                boxes = output["pred_boxes"][index]
                scores = output["scores"][index]
                classes = output["pred_classes"][index]
                results.append((boxes, scores, classes))
            else:
                results.append(None)
        return results

    def get_ret_from_prediction(self, predictions):
        boxes = predictions.pred_boxes if predictions.has(
            "pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has(
            "pred_classes") else None
        return {
            "pred_boxes": boxes.tensor.data.numpy(),
            "scores": scores.data.numpy(),
            "pred_classes": classes.data.numpy()
        }

    def __call__(self, images):
        return self.predict_one(images)


def get_clasp_dataset():
    """A dummy CLASP dataset that includes only the 'classes' field."""
    classes = ["bin_empty", "item"]
    # ds = {i: name for i, name in enumerate(classes)}
    ds = {0: "bin_empty", 1: "item"}
    return ds


def get_class_string(class_index, dataset):
    class_text = (dataset[class_index]
                  if dataset is not None else "id{:d}".format(class_index))
    return class_text


# def _create_text_labels(classes, scores, class_names):
#     """
#     Args:
#         classes (list[int] or None):
#         scores (list[float] or None):
#         class_names (list[str] or None):

#     Returns:
#         list[str] or None
#     """
#     labels = None
#     if classes is not None and class_names is not None and len(
#             class_names) > 0:
#         labels = [class_names[i] for i in classes]
#     if scores is not None:
#         if labels is None:
#             labels = ["{:.0f}%".format(s * 100) for s in scores]
#         else:
#             labels = [
#                 "{} {:.0f}%".format(l, s * 100)
#                 for l, s in zip(labels, scores)
#             ]
#     return labels


def vis_class(img, pos, class_str, font_scale=0.5):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0

    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img,
                txt,
                txt_tl,
                font,
                font_scale,
                _GRAY,
                lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, class_str=None, thick=2):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img


def vis_one_image_opencv(im,
                         boxes,
                         scores,
                         classes,
                         thresh=0.5,
                         dataset=None,
                         show_class=True,
                         vis=True):
    """Constructs a numpy array with the detections visualized."""

    if boxes is None or boxes.shape[0] == 0 or max(scores) < thresh:
        return im, None, None, None

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    _boxes = []
    _score = []
    _class = []

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores[i]
        if score < thresh:
            continue

        # show class (off by default)
        class_str = get_class_string(classes[i], dataset)
        if vis and show_class:
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        if vis:
            im = vis_bbox(
                im,
                (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                class_str,
            )

        _boxes.append(bbox)
        _score.append(score)
        _class.append(class_str)

    return im, np.array(_boxes), np.array(_score), np.array(_class)
