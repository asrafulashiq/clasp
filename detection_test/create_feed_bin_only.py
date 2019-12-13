# user-defined imports
# other imports
from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
from config import conf
from visutils.vis_feed import VisFeed
import glob, os
import pandas as pd
import skimage
import shutil
import pandas as pd
import visutils.vis as vis
from parse import parse
from collections import defaultdict


def to_sec(frame, fps=30):
    return str(int(frame) // fps)+'s'


class InfoClass:
    def __init__(self):
        bin_file = "./info/info.csv"

        bin_names = [
            "file",
            "camera",
            "frame",
            "id",
            "class",
            "x1",
            "y1",
            "x2",
            "y2",
            "type",
            "msg",
        ]
        self.df_bin = pd.read_csv(
            str(bin_file),
            sep=",",
            header=None,
            names=bin_names,
            index_col=None,
        )
        print("loaded")
        self.bin_pax = {}

    def get_info_fram_frame(self, frame, cam="cam09"):

        msglist = []
        # get bin info
        if frame % 2 == 0:
            _frame = frame + 1
        else:
            _frame = frame
        df = self.df_bin
        info = df[(df["frame"] == _frame - 50) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = "B" + str(row["id"])
                list_info_bin.append(
                    [
                        _id,
                        "item",
                        row["x1"],
                        row["y1"],
                        row["x2"],
                        row["y2"],
                        self.bin_pax.get(_id, ""),
                    ]
                )

            else:  # event type
                if row["frame"] != frame:
                    continue
                if row['type'] not in ('enter', 'exit'):
                    continue
                if row['type'] == 'enter' and row['camera'] == 'cam09':
                    continue
                list_event_bin.append([row["type"], row["msg"]])
                msglist.append(
                    [row["camera"][-2:], to_sec(row["frame"]), row["msg"]]
                )

        return (
            list_info_bin,
            [],
            list_event_bin,
            [],
            msglist
        )

    def draw_im(self, im, info_bin, info_pax, font_scale=0.5, cam='cam09'):
        for each_i in info_bin:
            bbox = [each_i[2], each_i[3], each_i[4], each_i[5]]

            im = vis.vis_bbox_with_str(
                im,
                bbox,
                each_i[0],
                each_i[-1],
                color=(33, 217, 14),
                thick=2,
                font_scale=font_scale,
                color_txt=(252, 3, 69)
            )
        return im


if __name__ == "__main__":

    file_num = "exp2_train"

    out_folder = {}
    imlist = []

    conf.skip_init = 2500
    conf.end_file = 11000
    conf.delta = 2

    feed_folder = Path(conf.out_dir) / "demo" / "feed_bin_only"
    if feed_folder.exists():
        shutil.rmtree(str(feed_folder))

    feed_folder.mkdir(exist_ok=True)

    vis_feed = VisFeed()

    Info = InfoClass()

    imlist = []
    src_folder = {}
    out_folder = {}

    # for cam in cameras:
    cameras = ['cam09', 'cam11']

    for cam in cameras:
        src_folder[cam] = Path(conf.root) / file_num / cam
        assert src_folder[cam].exists()

        imlist.append(
            utils.get_images_from_dir(
                src_folder[cam],
                skip_init=conf.skip_init,
                skip_end=conf.skip_end,
                delta=conf.delta,
                end_file=conf.end_file,
            )
        )

    for out1, out2 in tqdm(zip(*imlist)):
        im1, imfile1, _ = out1
        im2, imfile2, _ = out2

        frame_num = int(Path(imfile1).stem) - 1

        # draw image
        info_bin, info_pax, event_bin, event_pax, msglist = Info.get_info_fram_frame(
            frame_num, "cam09"
        )
        im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75, cam='cam09')

        info_bin, info_pax, event_bin, event_pax, mlist = Info.get_info_fram_frame(
            frame_num, "cam11"
        )
        im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7, cam='cam11')

        # get message
        msglist.extend(mlist)
        im_feed = vis_feed.draw(im1, im2, frame_num, msglist)

        f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
        skimage.io.imsave(str(f_write), im_feed)


