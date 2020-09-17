import torch
import torch.nn as nn
import torchvision
import argparse
import os
from torchvision.transforms import functional as F

import numpy as np
from pathlib import Path

import rcnn.model as models

import cv2
import skimage
from tqdm import tqdm


class RCNN_Detector():
    def __init__(self, ckpt=None, labels_to_keep=(2, 3), thres=0.5):
        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        self.model = models.get_model(num_classes=4)
        self.model.to(self.device)

        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(str(ckpt)))
        else:
            raise ValueError(f"{ckpt} does not exist")

        self.labels_to_keep = labels_to_keep
        self.thres = thres

    @torch.no_grad()
    def predict_one(self, im, show=False):
        im = skimage.img_as_float32(im)
        imt = F.to_tensor(im)
        self.model.eval()
        output = self.model([imt.to(self.device)])
        output = {k: v.cpu().data.numpy() for k, v in output[0].items()}
        index = (np.isin(output["labels"], self.labels_to_keep) &
                 (output["scores"] > self.thres))
        if index.size > 0:
            boxes = output["boxes"][index]
            scores = output["scores"][index]
            classes = output["labels"][index]
            return boxes, scores, classes
        else:
            return None

    # @torch.no_grad()
    # def predict_one(self, im, show=False):
    #     im = skimage.img_as_float32(im)
    #     imt = F.to_tensor(im)
    #     self.model.eval()
    #     output = self.model([imt.to(self.device)])
    #     output = {k: v.cpu().data.numpy() for k, v in output[0].items()}
    #     index = (np.isin(output["labels"], self.labels_to_keep) &
    #              (output["scores"] > self.thres))
    #     if index.size > 0:
    #         boxes = output["boxes"][index]
    #         scores = output["scores"][index]
    #         classes = output["labels"][index]
    #         return boxes, scores, classes
    #     else:
    #         return None

    @torch.no_grad()
    def predict_batch(self, imlist, show=False):
        imt = [F.to_tensor(im) for im in imlist]
        self.model.eval()
        imlist_gpu = torch.stack(imt, dim=0).to(self.device)
        batch_output = self.model(imlist_gpu)
        results = []
        for i in range(len(imlist)):
            output = {
                k: v.cpu().data.numpy()
                for k, v in batch_output[i].items()
            }
            index = (np.isin(output["labels"], self.labels_to_keep) &
                     (output["scores"] > self.thres))
            if index.size > 0:
                boxes = output["boxes"][index]
                scores = output["scores"][index]
                classes = output["labels"][index]
                results.append((boxes, scores, classes))
            else:
                results.append(None)
        return results

    def __call__(self, images):
        return self.predict_one(images)
