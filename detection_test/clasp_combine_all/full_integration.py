import sys
sys.path.append("../")

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
from colorama import init, Fore
from tools import read_nu_association, read_mu, read_rpi

init(autoreset=True)

############### Experiment name and cameras ###############
file_num = "exp2_train"
cameras = ["cam09", "cam11", "cam13"]

########### PAX and Bin detection files  ###########
bin_file = "./info/info.csv"

pax_file_9 = "./info/cam09exp2_logs_fullv1.txt"
pax_file_11 = "./info/cam11exp2_logs_fullv1.txt"
pax_file_13 = "./info/cam13exp2_logs_fullv1.txt"

nu_file_cam9 = "./info/cam_09_exp2_associated_events.csv"
nu_file_cam11 = "./info/cam_11_exp2_associated_events.csv"
nu_file_cam13 = "./info/cam_13_exp2_associated_events.csv"

# # NOTE: Travel Unit info for exp2_train. Make it empty if we don't know TU info.
TU_info = {}
# TU_info = {
#     'P2': 'TU1',
#     'P3': 'TU1',
#     'P4': 'TU1',
#     'P7': 'TU2',
#     'P12': 'TU3',
#     'P13': 'TU3',
#     'P14': 'TU3'
# }


def to_sec(frame, fps=30):
    # convert frame number to second
    return str(int(frame) // fps) + "s"


class IntegratorClass:
    def __init__(self):

        # load bin
        self.df_bin = self.load_bin()

        # load pax info
        self.df_pax = read_mu([pax_file_9, pax_file_11, pax_file_13],
                              scale=1. / 3)

        self.tmp_msg = []

        # get association info
        self.asso_info = {}
        self.asso_info["cam09"], self.asso_msg = self.get_association_info(
            nu_file_cam9, "09")
        self.asso_info["cam11"], asso_msg_11 = self.get_association_info(
            nu_file_cam11, "11")
        self.asso_info["cam13"], asso_msg_13 = self.get_association_info(
            nu_file_cam13, "13")
        self.asso_msg.update(asso_msg_11)
        self.asso_msg.update(asso_msg_13)

        print("loaded")
        self.bin_pax = {}  # association between pax and bin
        self.tmp = []

        # to determine first used
        self.item_first_used = []

    def load_bin(self):
        # load bin info
        df = read_rpi(bin_file, scale=None)

        # load change type
        loc_cng = df["type"] == "chng"
        df.loc[loc_cng, "frame"] = df[loc_cng]["frame"] - 50

        # sort by frame number
        df = df.sort_values("frame")
        return df

    def get_association_info(self, nu_file, cam="09"):
        asso_info = defaultdict(dict)  # store association info
        asso_msg = {}  # association message, like 'stealing'
        df_tmp = pd.read_csv(nu_file, header=None, names=["frame", "des"])

        for _, row in df_tmp.iterrows():
            frame = row["frame"]
            des = row["des"]
            des = parse("[{}]", des)
            if des is None:
                continue
            des = des[0]
            for each_split in des.split(","):
                each_split = each_split.strip()
                pp = parse("'P{}-B{}'", each_split)
                if pp is not None:
                    pax_id, bin_id = "P" + str(pp[0]), "B" + str(int(pp[1]))

                    #### potential theft info from NU ####
                    if ("stealing" in pax_id) or ("stoling" in pax_id):
                        pax_id = pax_id.replace("stoling", "stealing")
                        each_split = each_split.replace("stoling", "stealing")
                        if each_split not in self.tmp_msg:
                            asso_msg[frame] = [cam, frame, each_split]
                            self.tmp_msg.append(each_split)
                    else:
                        asso_info[bin_id][frame] = pax_id
        return asso_info, asso_msg

    def get_info_from_frame(self, frame, cam="cam09"):
        logs = []  # logs for evaluation

        # get pax info
        df = self.df_pax
        msglist = []
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_pax = []
        list_event_pax = []
        for _, row in info.iterrows():
            if row["type"] == "loc":
                list_info_pax.append([
                    row["id"], "pax", row["x1"], row["y1"], row["x2"],
                    row["y2"]
                ])

        # get bin info.
        df = self.df_bin
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []

        asso_info = self.asso_info["cam09"]  # association from camera 09
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = "B" + str(row["id"])  # convert id name to B[id]

                # get associated pax
                if _id in asso_info:
                    ffs = list(asso_info[_id])
                    for _f in ffs:
                        if frame >= _f:
                            self.bin_pax[_id] = asso_info[_id][_f]
                else:
                    pass

                # bin info for visualization
                list_info_bin.append([
                    _id,
                    "item",
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                    self.bin_pax.get(_id, ""),
                ])

            else:  # event type
                # Which event to skip
                if row["type"] not in ("enter", "exit"):
                    pass  # skip event
                elif row["type"] == "enter" and cam == "cam09":
                    # skip bin 'enter' event for cam 09
                    pass
                else:
                    list_event_bin.append([row["type"], row["msg"]])
                    msglist.append(
                        [row["camera"][-2:],
                         to_sec(row["frame"]), row["msg"]])

            # add 'theft' message for visualization
            if frame in self.asso_msg:
                rr = self.asso_msg[frame]
                if rr[0] == cam[3:5]:
                    if (cam + rr[2]) not in self.tmp:
                        msglist.append([rr[0], to_sec(rr[1]), rr[2]])
                        self.tmp.append(cam + rr[2])

        return (list_info_bin, list_info_pax, list_event_bin, list_event_pax,
                msglist)

    def draw_im(self, im, info_bin, info_pax, font_scale=0.5):
        # draw bin
        for each_i in info_bin:
            bbox = [each_i[2], each_i[3], each_i[4], each_i[5]]
            belongs_to = each_i[-1]

            # bin association to Travel Unit
            if each_i[-1] in TU_info:
                belongs_to = TU_info[each_i[-1]]
            im = vis.vis_bbox_with_str(
                im,
                bbox,
                each_i[0],
                belongs_to,
                color=(33, 217, 14),
                thick=2,
                font_scale=font_scale,
                color_txt=(252, 3, 69),
            )

        # draw pax
        for each_i in info_pax:
            bbox = [each_i[2], each_i[3], each_i[4], each_i[5]]

            # pax id to travel unit
            if each_i[0] in TU_info:
                tu_to = TU_info[each_i[0]]
            else:
                tu_to = None
            im = vis.vis_bbox_with_str(
                im,
                bbox,
                each_i[0],
                tu_to,
                color=(23, 23, 246),
                thick=2,
                font_scale=font_scale,
                color_txt=(252, 211, 3),
            )
        return im


if __name__ == "__main__":
    conf.plot = True
    out_folder = {}
    imlist = []

    vis_feed = VisFeed()  # Visualization class

    feed_folder = Path(conf.fmt_filename_out_feed.format(file_num=file_num))
    if feed_folder.exists():
        shutil.rmtree(str(feed_folder))
    feed_folder.mkdir(exist_ok=True)

    Info = IntegratorClass()  # integrator class info from 3 groups

    imlist = []
    src_folder = {}
    out_folder = {}

    for cam in cameras:
        src_folder[cam] = Path(
            conf.fmt_filename_src.format(file_num=file_num, cam=cam))
        assert src_folder[cam].exists()

        if cam == "cam13":
            start_frame = conf.start_frame - 50  # cam 13 lags by 50 frames from cam 09 and 11
        else:
            start_frame = conf.start_frame

        imlist.append(
            utils.get_images_from_dir(src_folder[cam],
                                      start_frame=start_frame,
                                      skip_end=conf.skip_end,
                                      delta=conf.delta,
                                      end_frame=conf.end_frame,
                                      file_only=False))

    pbar = tqdm(zip(*imlist))
    for out1, out2, out3 in pbar:
        im1, imfile1, _ = out1
        im2, imfile2, _ = out2
        im3, imfile3, _ = out3

        frame_num = int(Path(imfile1).stem) - 1

        pbar.set_description(f"Processing: {Fore.CYAN}{frame_num}")

        # Cam 09
        info_bin, info_pax, event_bin, event_pax, msglist = Info.get_info_from_frame(
            frame_num, "cam09")
        im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)

        # Cam 11
        info_bin, info_pax, event_bin, event_pax, mlist = Info.get_info_from_frame(
            frame_num, "cam11")
        msglist.extend(mlist)
        im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)

        # Cam 13
        frame_num3 = int(Path(imfile3).stem) - 1
        info_bin, info_pax, event_bin, event_pax, mlist = Info.get_info_from_frame(
            frame_num3, "cam13")
        msglist.extend(mlist)

        im3 = Info.draw_im(im3, info_bin, info_pax, font_scale=0.7)

        # News feed info
        # news-feed message
        im_feed = vis_feed.draw(im1, im2, im3, frame_num, msglist)

        f_write = feed_folder / (str(frame_num).zfill(6) + ".jpg")
        skimage.io.imsave(str(f_write), im_feed)
