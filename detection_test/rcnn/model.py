import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes=4, size=(640, 360)):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, min_size=min(size), max_size=max(size))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1)  # 1 for background
    return model