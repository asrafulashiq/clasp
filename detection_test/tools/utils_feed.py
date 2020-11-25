from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
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
# file_num = "exp2_train"
# cameras = ["cam09", "cam11", "cam13"]

# # PAX and Bin detection files
# bin_file = "./info/info_offset.csv"

# pax_file_9 = "./info/cam09exp2_logs_fullv1.txt"
# pax_file_11 = "./info/cam11exp2_logs_fullv1.txt"
# pax_file_13 = "./info/cam13exp2_logs_fullv1.txt"

# nu_file_cam9 = "./info/cam_09_exp2_associated_events.csv"
# nu_file_cam11 = "./info/cam_11_exp2_associated_events.csv"
# nu_file_cam13 = "./info/cam_13_exp2_associated_events.csv"


def to_sec(frame, fps=10):
    return str(int(frame) // fps) + "s"


def frame_to_time(imfile):
    # frame_num = int(Path(imfile1).stem) - 1
    frame_num = int(Path(imfile).stem.split("_")[-1])
    return frame_num


class IntegratorClass:
    def __init__(self, fps=10):
        self.bin_pax = {}
        self.tmp = []
        # to determine first used
        self.item_first_used = []
        self.fps = fps

    def load_info(self, bin_file, pax_file, nu_file_cam):
        # load bin
        self.df_bin = self.load_bin(bin_file)

        # load pax info
        self.df_pax = self.refine_pax_df(pax_file)

        # get association info
        self.asso_info = {}
        self.asso_info, _asso_msg = self.get_association_info(nu_file_cam)

        for k, v in _asso_msg.items():
            if k not in self.asso_info:
                self.asso_info[k] = {}
            self.asso_info[k].update(v)

        print("loaded")

    def load_bin(self, bin_file):
        # load bin info
        df = pd.read_csv(str(bin_file), sep=",")
        df["x1"] = df["x1"] / 3
        df["y1"] = df["y1"] / 3
        df["x2"] = df["x2"] / 3
        df["y2"] = df["y2"] / 3
        df["camera"] = df["cam"]
        # load location type
        df_new = df[df["type"] == "loc"].copy()
        df_comb = df_new

        # load change type
        # loc_cng = df["type"] == "chng"
        # df_comb.loc[loc_cng, "frame"] = df[loc_cng]["frame"] - 50
        # change detection offset
        df_comb = df_comb.sort_values("frame")
        return df_comb

    def refine_pax_df(self, pax_file):
        # pax_names = [
        #     "frame", "cam", "timeoffset", "x1", "y1", "x2", "y2", "id",
        #     "firstused"
        # ]
        df = pd.read_csv(str(pax_file), sep=",", index_col=0)
        df["x1"] = df["x1"] / (1080. / 640)
        df["y1"] = df["y1"] / (720. / 360)
        df["x2"] = df["x2"] / (1080. / 640)
        df["y2"] = df["y2"] / (720. / 360)
        df["camera"] = df["cam"].apply(lambda x: f"cam{int(x):02d}")
        # df["type"] = df["type"].str.lower()
        return df

    def get_association_info(self, nu_file):
        asso_info = defaultdict(lambda: defaultdict(dict))
        asso_msg = defaultdict(dict)
        # return asso_info, asso_msg

        # fid, cam, events
        df = pd.read_csv(nu_file, header=None, names=["frame", "cam", "des"])

        # def time2frame(time):
        #     time = str(time)
        #     sec, msec = time[:-2], time[-2:]
        #     frame = int(round(float(f"{sec}.{msec}") * 30))
        #     return frame

        # df['frame'] = df["time"].apply(time2frame)
        df['cam'] = df['cam'].apply(lambda x: f"cam{int(x[3:]):02d}")
        for _, row in df.iterrows():
            frame = row["frame"]
            des = row["des"]
            cam = row["cam"]
            des = parse("[{}]", des)
            if des is None:
                continue
            des = des[0]
            for each_split in des.split(","):
                each_split = each_split.strip()
                pp = parse(
                    "'{event_str}PP{pax_id:d}{tmp}Bin {bin_id:d}{_end}'",
                    each_split)
                if pp is not None:
                    bin_id, pax_id = 'B' + str(pp['bin_id']), 'P' + str(
                        pp['pax_id'])
                    if "XFR" in pp['event_str'] and "owner of" in pp['tmp']:
                        # association
                        asso_info[cam][bin_id][frame] = pax_id
                    elif "hand in" in pp['event_str']:
                        pass
                    elif "suspicious" in pp['event_str'].lower():
                        asso_msg[cam][frame] = [
                            cam, frame,
                            each_split.replace("|", "").replace("'", "")
                        ]
        return asso_info, asso_msg

    def get_info_from_frame(self, frame, cam="cam09"):
        # get pax info
        logs = []
        df = self.df_pax
        msglist = []
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_pax = []
        list_event_pax = []
        for _, row in info.iterrows():
            if not row.empty:  #row["type"] == "loc":
                row['id'] = 'P' + row['id']
                list_info_pax.append([
                    row["id"], "pax", row["x1"], row["y1"], row["x2"],
                    row["y2"]
                ])
                if "TSO" in row["id"]:
                    pax_type = "TSO"
                else:
                    pax_type = "PAX"
                log = (
                    f"LOC: type: {pax_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                    +
                    f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                    + f"ID: {row['id']}")
                logs.append(log)

        # get bin info.
        df = self.df_bin
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []

        asso_info = self.asso_info["cam09"]  # association from camera 09
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = str(row["id"])

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

                # FIXME left-behind calculation
                left_behind = "false"
                log = (
                    f"LOC: type: DVI camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                    +
                    f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                    +
                    f"ID: {_id} PAX-ID: {self.bin_pax.get(_id, 'NA')} left-behind: {left_behind}"
                )
                logs.append(log)

            else:  # event type
                if (row["type"] == "enter" and cam == "cam09") or \
                     row["type"] == "chng" or (row["type"] == "empty" and cam != "cam09"):
                    # xfr event
                    _id = "B" + str(row["id"])
                    _type = "TO" if cam == "cam09" else "FROM"
                    log = (
                        f"XFR: type: {_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                        +
                        f"BB: {row['x1']*3}, {row['y1']*3}, {row['x2']*3}, {row['y2']*3} "
                        +
                        f"PAX-ID: {self.bin_pax.get(_id, 'NA')} DVI-ID: {_id} theft: FALSE"
                    )
                    logs.append(log)
                    continue
                # Which event to skip
                if row["type"] not in ("enter", "exit"):
                    pass  # skip event
                elif row["type"] == "enter" and cam == "cam09":
                    # skip bin 'enter' event for cam 09
                    pass
                else:
                    list_event_bin.append([row["type"], row["msg"]])
                    msglist.append([
                        row["camera"][-2:],
                        to_sec(row["frame"], fps=self.fps), row["msg"]
                    ])

            # add 'theft' message for visualization
            # if frame in self.asso_msg:
            #     rr = self.asso_msg[frame]
            #     if rr[0] == cam[3:5]:
            #         if (cam + rr[2]) not in self.tmp:
            #             msglist.append([rr[0], to_sec(rr[1]), rr[2]])
            #             self.tmp.append(cam + rr[2])

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


class DrawClass():
    def __init__(self, conf=None, plot=True) -> None:
        self.vis_feed = VisFeed()
        self.Info = IntegratorClass(fps=conf.fps)
        self.plot = plot
        self.conf = conf

        # Output folder path of the feed
        self.feed_folder = Path(conf.out_dir) / "run" / conf.file_num / "feed"
        if self.feed_folder.exists():
            shutil.rmtree(str(self.feed_folder))
        self.feed_folder.mkdir(exist_ok=True)

        self.full_log = []

    def run_process(self, batch_frames, *csv_files):
        import multiprocessing

        def get_proc():
            return multiprocessing.Process(target=self.draw_batch,
                                           args=(batch_frames, *csv_files))

        if not hasattr(self, "proc"):
            self.proc = get_proc()

        if self.proc.is_alive():
            self.proc.join()
            # self.proc.terminate()
            self.proc.close()
            self.proc = get_proc()
        self.proc.start()

    def draw_batch(self, batch_frames, *csv_files):
        self.Info.load_info(*csv_files)
        # for out in tqdm(batch_frames, desc="Drawing", position=10):
        for out in batch_frames:
            self._draw_frame(*out)

    def _draw_frame(self, out1, out2, out3=None):
        im1, imfile1, _ = out1
        im2, imfile2, _ = out2
        if out3 is not None:
            im3, imfile3, _ = out3

        frame_num = frame_to_time(imfile1)
        # Cam 09
        info_bin, info_pax, event_bin, event_pax, msglist, logs = self.Info.get_info_from_frame(
            frame_num, "cam09")
        self.full_log.extend(logs)
        if self.plot:
            im1 = self.Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)

        # print("#draw:1", flush=True)
        # Cam 11
        info_bin, info_pax, event_bin, event_pax, mlist, logs = self.Info.get_info_from_frame(
            frame_num, "cam11")
        self.full_log.extend(logs)
        if self.plot:
            im2 = self.Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)

        # Cam 13
        im3 = None
        if out3 is not None:
            frame_num3 = frame_to_time(imfile3)
            info_bin, info_pax, event_bin, event_pax, mlist, logs = self.Info.get_info_from_frame(
                frame_num3, "cam13")
            self.full_log.extend(logs)
            if self.plot:
                im3 = self.Info.draw_im(im3,
                                        info_bin,
                                        info_pax,
                                        font_scale=0.7)

        # News feed info

        # print("#draw:2", flush=True)
        if self.plot:
            # get message
            msglist.extend(mlist)
            im_feed = self.vis_feed.draw(im1, im2, im3, frame_num, msglist)
            # print("#draw:2a", flush=True)

            f_write = self.feed_folder / (str(frame_num).zfill(4) + ".jpg")
            skimage.io.imsave(str(f_write), im_feed)
            # print("#draw:2b", flush=True)

        # print("#draw:3", flush=True)

    def finish(self):
        log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "..", "ata_logs")
        os.makedirs(log_folder, exist_ok=True)
        logpath = os.path.join(log_folder, self.conf.ata_out)
        with open(logpath, "w") as fp:
            fp.write("\n".join(self.full_log))


# if __name__ == "__main__":

#     out_folder = {}
#     imlist = []

#     if conf.plot:
#         vis_feed = VisFeed()  # Visualization class

#         # Output folder path of the feed
#         feed_folder = Path(conf.out_dir) / "run" / file_num / "feed"
#         if feed_folder.exists():
#             shutil.rmtree(str(feed_folder))
#         feed_folder.mkdir(exist_ok=True)

#     Info = IntegratorClass()  # integrate info from 3 groups

#     imlist = []
#     src_folder = {}
#     out_folder = {}

#     for cam in cameras:
#         src_folder[cam] = Path(conf.root) / file_num / cam
#         assert src_folder[cam].exists()

#         if cam == "cam13":
#             start_frame = conf.start_frame - 50  # cam 13 lags by 50 frames from cam 09 and 11
#         else:
#             start_frame = conf.start_frame

#         imlist.append(
#             utils.get_images_from_dir(src_folder[cam],
#                                       start_frame=start_frame,
#                                       skip_end=conf.skip_end,
#                                       delta=conf.delta,
#                                       end_frame=conf.end_frame,
#                                       file_only=(not conf.plot)))

#     full_log_cam09 = []
#     full_log_cam11 = []
#     full_log_cam13 = []

#     for out1, out2, out3 in tqdm(zip(*imlist)):
#         im1, imfile1, _ = out1
#         im2, imfile2, _ = out2
#         im3, imfile3, _ = out3

#         frame_num = int(Path(imfile1).stem) - 1

#         # Cam 09
#         info_bin, info_pax, event_bin, event_pax, msglist, logs = Info.get_info_from_frame(
#             frame_num, "cam09")
#         full_log_cam09.extend(logs)
#         if conf.plot:
#             im1 = Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)

#         # Cam 11
#         info_bin, info_pax, event_bin, event_pax, mlist, logs = Info.get_info_from_frame(
#             frame_num, "cam11")
#         full_log_cam11.extend(logs)
#         if conf.plot:
#             im2 = Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)

#         # Cam 13
#         frame_num3 = int(Path(imfile3).stem) - 1
#         info_bin, info_pax, event_bin, event_pax, mlist, logs = Info.get_info_from_frame(
#             frame_num3, "cam13")
#         full_log_cam13.extend(logs)
#         if conf.plot:
#             im3 = Info.draw_im(im3, info_bin, info_pax, font_scale=0.7)

#         # News feed info
#         if conf.plot:
#             # get message
#             msglist.extend(mlist)
#             im_feed = vis_feed.draw(im1, im2, im3, frame_num, msglist)

#             f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
#             skimage.io.imsave(str(f_write), im_feed)

#     # Write ata output file for scoring
#     with open("ata_cam9.txt", "w") as fp:
#         fp.write("\n".join(full_log_cam09))

#     with open("ata_cam11.txt", "w") as fp:
#         fp.write("\n".join(full_log_cam11))

#     with open("ata_cam13.txt", "w") as fp:
#         fp.write("\n".join(full_log_cam13))
