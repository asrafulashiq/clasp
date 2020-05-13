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
from tools import read_nu_association, read_mu, read_rpi

# PAX and Bin detection files
bin_file = "./info/info_exp2_cam09cam11cam13.csv"

pax_file_9 = "./info/cam09exp2_logs_full_may14.txt"
pax_file_11 = "./info/cam11exp2_logs_full_may14.txt"
pax_file_13 = "./info/cam13exp2_logs_full_may14.txt"

nu_file_cam9 = "./info/events_training_cam09exp2_102419.csv"
nu_file_cam11 = "./info/events_training_cam11exp2_102419.csv"
nu_file_cam13 = "./info/events_training_cam13exp2_102419.csv"

# Life of a bin and pax
BIN = "B27"
PAX = "P12"
conf.start_frame = 7260
conf.end_frame = 9800
conf.skip_end = None
conf.delta = 1

_BIN_COLOR = (33, 217, 14)
_BIN_COLOR_RED = (217, 17, 14)

current_color_bin = _BIN_COLOR
bin_start_frame = -1

PERSON_LEFT_BEHIND = 9440


def to_sec(frame, fps=30):
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
        self.bin_pax = {}
        self.tmp = []

        # to determine first used
        self.item_first_used = []

    def load_bin(self):
        # load bin info
        df = read_rpi(bin_file, scale=None)

        # load change type
        loc_cng = df["type"] == "chng"
        df.loc[loc_cng, "frame"] = df[loc_cng]["frame"] - 50

        # df["frame"] = df[
        #     "frame"] + 50  # FIXME: For some reason, there is a 50 frame lag

        # sort by frame number
        df = df.sort_values("frame")
        return df

    def get_association_info(self, nu_file, cam="09"):
        if nu_file is None:
            return {}, {}

        asso_info = defaultdict(dict)
        asso_msg = {}  # association message, like 'stealing'

        df = pd.read_csv(str(nu_file), header=None, names=["frame", "des"])

        for _, row in df.iterrows():
            frame = row["frame"]
            des = row["des"]
            des = parse("[{}]", des)
            if des is None:
                continue
            des = des[0]
            for each_split in des.split(","):
                each_split = each_split.strip()
                pp = parse("'|P{pax_id:d}|{event_str}|Bin {bin_id:d}|'",
                           each_split)
                if pp is not None:
                    bin_id, pax_id = 'B' + str(pp['bin_id']), 'P' + str(
                        pp['pax_id'])
                    if "owner of" in pp['event_str']:
                        # association
                        asso_info[bin_id][frame] = pax_id
                    elif "hand in" in pp['event_str']:
                        pass
                    elif "suspicious" in pp['event_str']:
                        asso_msg[frame] = [
                            cam, frame,
                            each_split.replace("|", "").replace("'", "")
                        ]
        return asso_info, asso_msg

    def get_info_from_frame(self, frame, cam="cam09"):

        # get pax info
        df = self.df_pax
        msglist = []
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_pax = []
        list_event_pax = []

        line_pt = [None, None]
        for _, row in info.iterrows():
            if row["type"] == "loc":
                if row["id"] != PAX:
                    continue
                list_info_pax.append([
                    row["id"], "pax", row["x1"], row["y1"], row["x2"],
                    row["y2"]
                ])

                global bin_start_frame
                if bin_start_frame == -1 and cam == "cam13":
                    bin_start_frame = frame

                #! SPECIFIC to LIFE_OF_A_BIN
                cx = int((float(row['x1']) + float(row['x2'])) / 2)
                cy = int((float(row['y1']) + float(row['y2'])) / 2)
                line_pt[0] = (cx, cy)
        # get bin info
        # Generally, bins are extracted for each odd frame
        df = self.df_bin

        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []

        asso_info = self.asso_info["cam09"]
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = str(row["id"])
                if _id != BIN:
                    continue
                if _id in asso_info:
                    ffs = list(asso_info[_id])
                    for _f in ffs:
                        if frame >= _f:
                            self.bin_pax[_id] = asso_info[_id][_f]
                else:
                    pass
                if _id in self.item_first_used:
                    self.item_first_used.append(_id)
                list_info_bin.append([
                    _id,
                    "item",
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                    self.bin_pax.get(_id, ""),
                ])

                #! SPECIFIC to LIFE_OF_A_BIN
                cx = int((float(row['x1']) + float(row['x2'])) / 2)
                cy = int((float(row['y1']) + float(row['y2'])) / 2)
                line_pt[1] = (cx, cy)

            else:  # event type
                if row["type"] not in ("enter", "exit"):
                    continue
                list_event_bin.append([row["type"], row["msg"]])
                msglist.append(
                    [row["camera"][-2:],
                     to_sec(row["frame"]), row["msg"]])

            if frame in self.asso_msg:
                rr = self.asso_msg[frame]
                if rr[2] not in self.tmp:
                    msglist.append([rr[0], to_sec(rr[1]), rr[2]])
                    self.tmp.append(rr[2])

        return (list_info_bin, list_info_pax, list_event_bin, list_event_pax,
                msglist, line_pt)

    def draw_im(self, im, info_bin, info_pax, font_scale=0.5):
        for each_i in info_bin:
            bbox = [each_i[2], each_i[3], each_i[4], each_i[5]]
            im = vis.vis_bbox_with_str(
                im,
                bbox,
                each_i[0],
                each_i[-1],
                color=current_color_bin,
                thick=2,
                font_scale=font_scale,
                color_txt=(252, 3, 69),
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
                color_txt=(252, 211, 3),
            )
        return im


if __name__ == "__main__":

    conf.plot = True
    out_folder = {}
    imlist = []

    vis_feed = VisFeed()  # Visualization class

    feed_folder = Path((conf.fmt_filename_out_feed +
                        "_left_behind_s").format(file_num=conf.file_num))
    if feed_folder.exists():
        shutil.rmtree(str(feed_folder))
    feed_folder.mkdir(exist_ok=True, parents=True)

    Info = IntegratorClass()  # integrator class info from 3 groups

    imlist = []
    src_folder = {}
    out_folder = {}

    for cam in conf.cameras:
        src_folder[cam] = Path(
            conf.fmt_filename_src.format(file_num=conf.file_num, cam=cam))
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
                                      fmt=conf.fmt,
                                      file_only=False))

    for out1, out2, out3 in tqdm(zip(*imlist)):
        im1, imfile1, frame_num1 = out1
        im2, imfile2, frame_num2 = out2
        im3, imfile3, frame_num3 = out3

        # Cam 09
        info_bin, info_pax, event_bin, event_pax, msglist, lpt = Info.get_info_from_frame(
            frame_num1, "cam09")
        if conf.plot:
            im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)
            if lpt[0] is not None and lpt[1] is not None:
                cv2.line(im1, lpt[0], lpt[1], (235, 164, 52), thickness=3)
                cv2.circle(im1, lpt[0], 3, (255, 0, 0), -1)
                cv2.circle(im1, lpt[1], 3, (255, 0, 0), -1)

        # Cam 11
        info_bin, info_pax, event_bin, event_pax, mlist, lpt = Info.get_info_from_frame(
            frame_num2, "cam11")
        if conf.plot:
            im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)
            if lpt[0] is not None and lpt[1] is not None:
                cv2.line(im2, lpt[0], lpt[1], (235, 164, 52), thickness=3)
                cv2.circle(im2, lpt[0], 3, (255, 0, 0), -1)
                cv2.circle(im2, lpt[1], 3, (255, 0, 0), -1)

        # Cam 13
        info_bin, info_pax, event_bin, event_pax, mlist, lpt = Info.get_info_from_frame(
            frame_num3, "cam13")
        if conf.plot:
            if bin_start_frame != -1 and (frame_num3 - bin_start_frame) > 900:
                if frame_num3 % 30 == 0:
                    # color switch
                    if current_color_bin == _BIN_COLOR:
                        current_color_bin = _BIN_COLOR_RED
                    elif current_color_bin == _BIN_COLOR_RED:
                        current_color_bin = _BIN_COLOR

            im3 = Info.draw_im(im3, info_bin, info_pax, font_scale=0.7)
            if lpt[0] is not None and lpt[1] is not None:
                cv2.line(im3, lpt[0], lpt[1], (235, 164, 52), thickness=3)
                cv2.circle(im3, lpt[0], 3, (255, 0, 0), -1)
                cv2.circle(im3, lpt[1], 3, (255, 0, 0), -1)

            # plot red boundary to denote timer

        # News feed info
        if conf.plot:
            # get message
            msglist.extend(mlist)

            if frame_num1 > 9440:
                sec_pass = (frame_num1 - 9440) / 30
                msglist = [["", "", ""],
                           [
                               "11",
                               to_sec(frame_num1),
                               f"{PAX} out of view for {sec_pass:.2f} sec"
                           ], ["11",
                               to_sec(frame_num1), f"{BIN} left behind"]]

            if frame_num1 >= (242 * 30) and frame_num1 < (268 * 30):
                ims = [im1, None, None]
            elif frame_num1 >= (268 * 30) and frame_num1 < (297 * 30):
                ims = [None, im2, None]
            elif frame_num1 >= (297 * 30):
                ims = [None, None, im3]

            im_feed = vis_feed.draw(*ims, frame_num1, msglist, with_feed=False)

            f_write = feed_folder / (str(frame_num1).zfill(6) + ".jpg")
            skimage.io.imsave(str(f_write), im_feed)
