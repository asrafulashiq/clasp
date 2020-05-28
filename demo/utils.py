
import cv2
import os
import sys
import time
import numpy as np
import skimage


_GRAY = (218, 227, 218)

def vis_bbox(img, bbox, color=(0,255,0), thick=2):
    """Visualizes a bounding box."""
    img = skimage.img_as_ubyte(img)
    (x0, y0, x1, y1) = bbox
    x1, y1 = int(x1), int(y1)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def vis_class(img, pos, class_str, color=(0, 255, 0), font_scale=1, thickness=2):
    """Visualizes the class."""
    img = skimage.img_as_ubyte(img)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 5)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0

    cv2.rectangle(img, back_tl, back_br, color, thickness)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale,
                color, lineType=cv2.LINE_AA, thickness=thickness)
    return img

def vis_box_class(img, bbox, class_str, color=(0, 255, 0), thick=0.5):
    img = vis_bbox(img, bbox, color, thick=thick)
    img = vis_class(img, bbox, class_str, color, thickness=thick)

    return img