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

# Experiment
file_num = "exp2_train"
cameras = ["cam09", "cam11", "cam13"]

# PAX and Bin detection files
bin_file = "./info/info_offset.csv"

pax_file_9 = "./info/cam09exp2_logs_fullv1.txt"
pax_file_11 = "./info/cam11exp2_logs_fullv1.txt"
pax_file_13 = "./info/cam13exp2_logs_fullv1.txt"

nu_file_cam9 = "./info/cam_09_exp2_associated_events.csv"
nu_file_cam11 = "./info/cam_11_exp2_associated_events.csv"
nu_file_cam13 = "./info/cam_13_exp2_associated_events.csv"


def to_sec(frame, fps=30):
    return str(int(frame) // fps) + "s"


class IntegratorClass:
    def __init__(self):

        # load bin
        self.df_bin = self.load_bin()

        # load pax info
        pax_names = [
            "frame", "id", "x1", "y1", "x2", "y2", "cam", "TU", "type"
        ]
        df_pax_9 = pd.read_csv(str(pax_file_9),
                               sep=",",
                               header=None,
                               names=pax_names,
                               index_col=None)
        df_pax_11 = pd.read_csv(str(pax_file_11),
                                sep=",",
                                header=None,
                                names=pax_names,
                                index_col=None)
        df_pax_13 = pd.read_csv(str(pax_file_13),
                                sep=",",
                                header=None,
                                names=pax_names,
                                index_col=None)
        self.tmp_msg = []

        self.df_pax = pd.concat((df_pax_9, df_pax_11, df_pax_13))
        self.df_pax = self.refine_pax_df()

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
        df = pd.read_csv(str(bin_file),
                         sep=",",
                         header=None,
                         names=bin_names,
                         index_col=None)

        # load location type
        df_new = df[df["type"] == "loc"].copy()
        df_new["frame"] = df["frame"] + 1
        df_comb = pd.concat((df, df_new))

        # load change type
        loc_cng = df["type"] == "chng"
        df_comb.loc[loc_cng, "frame"] = df[loc_cng]["frame"] - 50
        # change detection offset
        df_comb = df_comb.sort_values("frame")
        return df_comb

    def refine_pax_df(self):
        df = self.df_pax
        df["x1"] = df["x1"] / 3
        df["y1"] = df["y1"] / 3
        df["x2"] = df["x2"] / 3
        df["y2"] = df["y2"] / 3
        df["camera"] = df["cam"].apply(lambda x: x[:5])
        df["type"] = df["type"].str.lower()
        return df

    def get_association_info(self, nu_file, cam="09"):
        asso_info = defaultdict(dict)
        asso_msg = {}
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
                if pp is not None:
                    pax_id, bin_id = "P" + str(pp[0]), "B" + str(int(pp[1]))
                    if "stealing" in pax_id:
                        if each_split not in self.tmp_msg:
                            asso_msg[frame] = [cam, frame, each_split]
                            self.tmp_msg.append(each_split)
                    else:
                        asso_info[bin_id][frame] = pax_id
        return asso_info, asso_msg

    def get_info_from_frame(self, frame, cam="cam09"):

        logs = []

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
                first_used = False
                if row["id"] in self.item_first_used:
                    first_used = True
                    self.item_first_used.append(row["id"])
                if "TSO" in row["id"]:
                    pax_type = "TSO"
                else:
                    pax_type = "PAX"
                log = (
                    f"LOC: type: {pax_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/30:.2f} "
                    +
                    f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                    + f"ID: {row['id']} PAX-ID: NA first-used: {first_used} " +
                    "partial-complete: NA")
                logs.append(log)

        # get bin info
        # Generally, bins are extracted for each odd frame
        _frame = frame
        df = self.df_bin

        info = df[(df["frame"] == _frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []

        asso_info = self.asso_info[cam]
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = "B" + str(row["id"])
                if _id in asso_info:
                    ffs = list(asso_info[_id])
                    for _f in ffs:
                        if frame >= _f:
                            self.bin_pax[_id] = asso_info[_id][_f]
                else:
                    pass
                first_used = False
                if _id in self.item_first_used:
                    first_used = True
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
                log = (
                    f"LOC: type: DVI camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/30:.2f} "
                    +
                    f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                    +
                    f"ID: {_id} PAX-ID: {self.bin_pax.get(_id, 'NA')} first-used: {first_used} "
                    + "partial-complete: NA")
                logs.append(log)
            else:  # event type
                if (row["type"] == "enter" and cam == "cam09") or \
                     row["type"] == "chng" or (row["type"] == "empty" and cam != "cam09"):
                    # xfr event
                    _id = "B" + str(row["id"])
                    _type = "TO" if cam == "cam09" else "FROM"
                    log = (
                        f"XFR: type: {_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/30:.2f} "
                        +
                        f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                        +
                        f"owner-ID: {self.bin_pax.get(_id, 'NA')} DVI-ID: {_id} theft: False"
                    )
                    logs.append(log)
                    continue

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
                msglist, logs)

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

    out_folder = {}
    imlist = []

    if conf.plot:
        vis_feed = VisFeed()  # Visualization class

        # Output folder path of the feed
        feed_folder = Path(conf.out_dir) / "run" / file_num / "feed"
        if feed_folder.exists():
            shutil.rmtree(str(feed_folder))
        feed_folder.mkdir(exist_ok=True)

    Info = IntegratorClass()  # integrate info from 3 groups

    imlist = []
    src_folder = {}
    out_folder = {}

    for cam in cameras:
        src_folder[cam] = Path(conf.root) / file_num / cam
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
                                      file_only=(not conf.plot)))

    full_log_cam09 = []
    full_log_cam11 = []
    full_log_cam13 = []

    for out1, out2, out3 in tqdm(zip(*imlist)):
        im1, imfile1, _ = out1
        im2, imfile2, _ = out2
        im3, imfile3, _ = out3

        frame_num = int(Path(imfile1).stem) - 1

        # Cam 09
        info_bin, info_pax, event_bin, event_pax, msglist, logs = Info.get_info_from_frame(
            frame_num, "cam09")
        full_log_cam09.extend(logs)
        if conf.plot:
            im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)

        # Cam 11
        info_bin, info_pax, event_bin, event_pax, mlist, logs = Info.get_info_from_frame(
            frame_num, "cam11")
        full_log_cam11.extend(logs)
        if conf.plot:
            im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)

        # Cam 13
        frame_num3 = int(Path(imfile3).stem) - 1
        info_bin, info_pax, event_bin, event_pax, mlist, logs = Info.get_info_from_frame(
            frame_num3, "cam13")
        full_log_cam13.extend(logs)
        if conf.plot:
            im3 = Info.draw_im(im3, info_bin, info_pax, font_scale=0.7)

        # News feed info
        if conf.plot:
            # get message
            msglist.extend(mlist)
            im_feed = vis_feed.draw(im1, im2, im3, frame_num, msglist)

            f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
            skimage.io.imsave(str(f_write), im_feed)

    # Write ata output file for scoring
    with open("ata_cam9.txt", "w") as fp:
        fp.write("\n".join(full_log_cam09))

    with open("ata_cam11.txt", "w") as fp:
        fp.write("\n".join(full_log_cam11))

    with open("ata_cam13.txt", "w") as fp:
        fp.write("\n".join(full_log_cam13))
