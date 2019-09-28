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
    return int(frame) // fps


class InfoClass:
    def __init__(self):
        bin_file = "./info/info.csv"
        pax_file_9 = "./info/cam09exp2_logs_full_seg.txt"
        pax_file_11 = "./info/cam11exp2_logs_full_seg.txt"

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

        pax_names = ["frame", "id", "x1", "y1", "w", "h", "cam", "TU", "type"]

        df_pax_9 = pd.read_csv(
            str(pax_file_9),
            sep=",",
            header=None,
            names=pax_names,
            index_col=None,
        )

        df_pax_11 = pd.read_csv(
            str(pax_file_11),
            sep=",",
            header=None,
            names=pax_names,
            index_col=None,
        )

        self.df_pax = pd.concat((df_pax_9, df_pax_11))
        self.df_pax = self.refine_pax_df()
        self.get_association_info()

        print("loaded")
        self.bin_pax = {}

    def refine_pax_df(self):
        df = self.df_pax
        df["x1"] = df["x1"] / 3
        df["y1"] = df["y1"] / 3
        df["x2"] = df["x1"] + df["w"] / 3 - 1
        df["y2"] = df["y1"] + df["h"] / 3 - 1
        df["camera"] = df["cam"].apply(lambda x: x[:5])
        df["type"] = df["type"].str.lower()
        return df

    def get_association_info(self):
        self.dict_association = defaultdict(dict)
        nu_file = "./info/cam_09_exp2_associated_events.csv"
        df_tmp = pd.read_csv(nu_file, header=None, names=["frame", "des"])

        for _, row in df_tmp.iterrows():
            frame = row["frame"]
            des = row["des"]
            des = parse("[{}]", des)
            if des is None:
                continue
            des = des[0]
            for each_split in des.split(","):
                pp = parse("'P{}-B{}'", each_split)
                if pp is None:
                    continue
                pax_id, bin_id = "P" + str(pp[0]), "B" + str(int(pp[1]) - 1)
                self.dict_association[frame][bin_id] = pax_id

    def get_info_fram_frame(self, frame, cam="cam09"):

        # get pax info
        df = self.df_pax
        msglist = []
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_pax = []
        list_event_pax = []
        for _, row in info.iterrows():
            if row["type"] == "loc":
                list_info_pax.append(
                    [
                        row["id"],
                        "pax",
                        row["x1"],
                        row["y1"],
                        row["x2"],
                        row["y2"],
                    ]
                )

        # get bin info
        if frame % 2 == 0:
            _frame = frame + 1
        else:
            _frame = frame
        df = self.df_bin
        info = df[(df["frame"] == _frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = "B" + str(row["id"])
                if (
                    frame in self.dict_association
                    and _id in self.dict_association[frame]
                ):
                    self.bin_pax[_id] = self.dict_association[frame][_id]
                else:
                    pass
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
                list_event_bin.append([row["type"], row["msg"]])
                msglist.append(
                    [row["camera"][-2:], to_sec(row["frame"]), row["msg"]]
                )

        return (
            list_info_bin,
            list_info_pax,
            list_event_bin,
            list_event_pax,
            msglist,
        )

    def draw_im(self, im, info_bin, info_pax, font_scale=0.5):
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
            )

        for each_i in info_pax:
            bbox = [each_i[2], each_i[3], each_i[4], each_i[5]]
            im = vis.vis_bbox_with_str(
                im,
                bbox,
                each_i[0],
                None,
                color=(23, 23, 246),
                thick=2,
                font_scale=font_scale,
            )
        return im


if __name__ == "__main__":

    file_num = "exp2"
    cameras = ["cam09", "cam11"]

    out_folder = {}
    imlist = []

    feed_folder = Path(conf.out_dir) / "run" / file_num / "feed"
    if feed_folder.exists():
        shutil.rmtree(str(feed_folder))

    feed_folder.mkdir(exist_ok=True)

    vis_feed = VisFeed()

    Info = InfoClass()

    imlist = []
    src_folder = {}
    out_folder = {}

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
        im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)

        info_bin, info_pax, event_bin, event_pax, mlist = Info.get_info_fram_frame(
            frame_num, "cam11"
        )
        im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)

        # get message
        msglist.extend(mlist)
        im_feed = vis_feed.draw(im1, im2, frame_num, msglist)

        f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
        skimage.io.imsave(str(f_write), im_feed)

    cv2.destroyAllWindows()

