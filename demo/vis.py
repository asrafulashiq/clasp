from pathlib import Path
import cv2
from tqdm import tqdm
import skimage
import skimage.io as io
import os
import pandas as pd
import numpy as np
import utils
from tqdm import tqdm
import sys
import time



_GRAY = (218, 227, 218)

def vis_bbox(img, bbox, color=(0,255,0), thick=2):
    """Visualizes a bounding box."""
    img = skimage.img_as_ubyte(img)
    (x0, y0, x1, y1) = bbox
    x1, y1 = int(x1), int(y1)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def vis_class(img, pos, class_str, color=(0, 255, 0), font_scale=1):
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

    cv2.rectangle(img, back_tl, back_br, color, 2)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale,
                color, lineType=cv2.LINE_AA, thickness=2)
    return img

def vis_box_class(img, bbox, class_str, color=(0, 255, 0)):
    img = vis_bbox(img, bbox, color)
    img = vis_class(img, bbox, class_str)

    return img


def get_info_fram_frame(df, frame, cam="cam09"):
    if frame % 2 == 0:
        frame += 1
    info = df[(df["frame"] == frame) & (df["camera"] == cam)]
    list_info = []
    for _, row in info.iterrows():
        list_info.append(
            [
                row["id"],
                row["class"],
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"],
            ]
        )
    return list_info


if __name__ == "__main__":
    
    HOME = Path(os.environ["HOME"])

    # folder where raw frames are stored
    folder_frames = HOME / "dataset/ALERT/alert_frames/exp2/" 

    # csv file path
    info_file = HOME / "dataset/ALERT/clasp_data/output/run/info.csv"

    # folder where output image should be stored
    out_folder = HOME / "dataset/ALERT/clasp_data/output/demo/tmp"
    out_folder.mkdir(exist_ok=True, parents=True)



    # which camera to use
    cam = "cam11"

    # read csv file into pandas dataframe
    df = pd.read_csv(
        str(info_file),
        sep=",",
        header=None,
        names=[
            "file",
            "camera",
            "frame",
            "id",
            "class",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
        index_col=None,
    )

    # frames to work on
    frames = np.arange(2401, 2501)

    for frame in tqdm(frames):
        info = get_info_fram_frame(df, frame, cam)
        img = io.imread(str(folder_frames / cam / f"{frame:06d}.jpg"))
        img = skimage.transform.resize(
            img, size[::-1], anti_aliasing=False, order=1
        )

        for each_i in info:
            _id, cls, x1, y1, x2, y2 = each_i
            cls_str = f"{_id}"
            img = vis_box_class(img, (x1, y1, x2, y2),
                cls_str)
        
        out_file = out_folder / f"{frame:06d}.jpg"
        skimage.io.imsave(str(out_file), img)