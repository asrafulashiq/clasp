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
    folder_run = HOME / "dataset/ALERT/clasp_data/output/run/"
    folder_frames = HOME / "dataset/ALERT/alert_frames/exp2/"
    info_file = folder_run / "info.csv"

    out_folder = HOME / "dataset/ALERT/clasp_data/output/demo" \
        / "tso_2"
    out_folder.mkdir(exist_ok=True, parents=True)

    cam = "cam11"
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

    print(get_info_fram_frame(df, 5617))

    # START
    frames = np.arange(2985, 3453)
    size = (640, 360)

    for frame in tqdm(frames):
        info = get_info_fram_frame(df, frame, cam)
        img = io.imread(str(folder_frames / cam / f"{frame:06d}.jpg"))
        img = skimage.transform.resize(
            img, size[::-1], anti_aliasing=False, order=1
        )

        for each_i in info:
            _id, cls, x1, y1, x2, y2 = each_i
            cls_str = f"{_id}"
            img = utils.vis_box_class(img, (x1, y1, x2, y2),
                cls_str)
        
        out_file = out_folder / f"{frame:06d}.jpg"
        skimage.io.imsave(str(out_file), img)