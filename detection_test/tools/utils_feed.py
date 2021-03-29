import multiprocessing
from pathlib import Path
import cv2
from numpy.core import multiarray
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
        self.asso_info = defaultdict(lambda: defaultdict(dict))

        self.msg = set()

    def load_info(self, bin_file, pax_file, nu_file_cam):
        # load bin
        self.df_bin = self.load_bin(bin_file)

        # load pax info
        self.df_pax = self.refine_pax_df(pax_file)

        # get association info

        asso_info, self._asso_msg = self.get_association_info(nu_file_cam)
        # for k, v in _asso_msg.items():
        #     if k not in self.asso_info:
        #         asso_info[k] = {}
        #     asso_info[k].update(v)

        # for cam, _v1 in asso_info.items():
        #     for bin_id, _v2 in _v1.items():
        #         for fnum, pax_id in _v2.items():
        #             self.asso_info[cam][bin_id][fnum] = pax_id

        print("loaded")

    def load_bin(self, bin_file):
        # load bin info
        df = pd.read_csv(str(bin_file), sep=",")
        df["x1"] = df["x1"] / 3
        df["y1"] = df["y1"] / 3
        df["x2"] = df["x2"] / 3
        df["y2"] = df["y2"] / 3
        df["camera"] = df["cam"]
        df['id'] = 'B' + df['id'].astype(str)
        # load location type
        # df_new = df[df["type"] == "loc"].copy()
        # df_comb = df_new

        # load change type
        # loc_cng = df["type"] == "chng"
        # df_comb.loc[loc_cng, "frame"] = df[loc_cng]["frame"] - 50
        # change detection offset
        df = df.sort_values("frame")
        return df

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
        return df

    def get_association_info(self, nu_file):
        asso_msg = defaultdict(dict)

        # fid, cam, events
        df = pd.read_csv(nu_file, header=None, names=["frame", "cam", "des"])
        df['cam'] = df['cam'].apply(lambda x: f"cam{int(x[3:]):02d}")
        for _, row in df.iterrows():
            frame = row["frame"]
            des = row["des"]
            cam = row["cam"]
            des = parse("[{}]", des)
            if des is None:
                continue
            des = des[0]
            des_splits = des.split("', '")
            for i, each_split in enumerate(des_splits):
                each_split = each_split.strip().replace("'",
                                                        "").replace("'", "")

                if 'XFR' in each_split:
                    _msg = parse(("XFR: type: {} camera-num: {} frame: {} "
                                  "time-offset: {} BB: {} PAX-ID: {pax_id} "
                                  "DVI-ID: {bin_id} theft: {}"), each_split)
                    asso_msg[cam][frame] = [
                        'XFR', cam, frame, _msg["pax_id"], _msg["bin_id"],
                        each_split.replace("[", "").replace("]", "")
                    ]
                elif 'Association' in each_split:
                    # 1156,cam9,"['Association | P1 | owner of | Bin 3 |']"
                    pp = parse(
                        "{event_str}P{pax_id:d}{tmp}Bin {bin_id:d}{_end}",
                        each_split)
                    if pp is not None:
                        bin_id, pax_id = 'B' + str(pp['bin_id']), 'P' + str(
                            pp['pax_id'])
                        if "owner of" in pp['tmp']:
                            # association
                            self.asso_info[cam][bin_id][frame] = pax_id
        return self.asso_info, asso_msg

    def get_info_from_frame(self, frame, cam="cam09"):
        # get pax info
        logs = []
        df = self.df_pax
        msglist = []
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_pax = []
        list_event_pax = []
        for _, row in info.iterrows():
            if not row.empty:
                row['id'] = str(row['id'])
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
                    f"BB: {int(row['x1']*3)}, {int(row['y1']*3)}, {int(row['x2']*3)}, {int(row['y2']*3)} "
                    + f"ID: {row['id']}")
                logs.append(log)

        # get bin info.
        df = self.df_bin
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info_bin = []
        list_event_bin = []

        asso_info = self.asso_info[cam]  # association from camera 09
        for _, row in info.iterrows():
            if row["type"] == "loc":
                _id = str(row["id"])

                # get associated pax
                if _id in asso_info:
                    ffs = list(asso_info[_id])
                    for _f in ffs:
                        if frame >= _f:
                            self.bin_pax[_id] = str(asso_info[_id][_f])
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
                left_behind = "FALSE"
                log = (
                    f"LOC: type: DVI camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                    +
                    f"BB: {int(row['x1']*3)}, {int(row['y1']*3)}, {int(row['x2']*3)}, {int(row['y2']*3)} "
                    +
                    f"ID: {_id} PAX-ID: {self.bin_pax.get(_id, 'NA')} left-behind: {left_behind}"
                )
                logs.append(log)
            else:  # event type
                if (row["type"] == "enter" and cam == "cam09") or \
                     row["type"] == "chng" or (row["type"] == "empty" and cam != "cam09"):
                    # xfr event
                    _id = str(row["id"])
                    _type = "TO" if cam == "cam09" else "FROM"
                    log = (
                        f"XFR: type: {_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                        +
                        f"BB: {int(row['x1']*3)}, {int(row['y1']*3)}, {int(row['x2']*3)}, {int(row['y2']*3)} "
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

        if cam in self._asso_msg and frame in self._asso_msg[cam]:
            mtype, cam, frame, pax_id, bin_id, _msg = self._asso_msg[cam][
                frame]
            if mtype == 'XFR':
                # _type = "TO" if cam == "cam09" else "FROM"
                # log = (
                #     f"XFR: type: {_type} camera-num: {cam[3:5]} frame: {frame} time-offset: {frame/self.fps:.2f} "
                #     + f"BB: 0, 0, 0, 0 " +
                #     f"PAX-ID: {pax_id} DVI-ID: {bin_id} theft: FALSE")
                logs.append(_msg)

                if 'theft: true' in _msg.lower():
                    feed_msg = f'Potential theft: {pax_id} - {bin_id}'
                else:
                    feed_msg = (f'XFR : {pax_id} - {bin_id}')
                if feed_msg not in self.msg:
                    msglist.append(feed_msg)
                    self.msg.add(feed_msg)

        return (list_info_bin, list_info_pax, list_event_bin, list_event_pax,
                msglist, logs)

    @staticmethod
    def draw_im(im, info_bin, info_pax, font_scale=0.5):
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

    def draw_batch(self, batch_frames, *csv_files):
        self.Info.load_info(*csv_files)
        # for out in tqdm(batch_frames, desc="Drawing", position=10):
        all_out_args = []
        for out in batch_frames:
            _args = self._extract_info_for_draw_frame(*out)
            all_out_args.append(_args)

        if self.plot:
            self.plot_fn(all_out_args, self.feed_folder)

    def plot_fn(self, all_out_args, feed_folder):
        if self.conf.workers_plot > 1:
            with multiprocessing.Pool(
                    processes=self.conf.workers_plot) as pool:
                for each_step_arg in all_out_args:
                    # DrawClass.plot_fun_step(each_step_arg, feed_folder)
                    pool.apply_async(
                        DrawClass.plot_fun_step,
                        (each_step_arg, feed_folder),
                        #  callback=lambda x: print("#"),
                        error_callback=lambda e: print("error"))
                pool.close()
                pool.join()
        else:
            for each_step_arg in all_out_args:
                DrawClass.plot_fun_step(each_step_arg, feed_folder)

    @staticmethod
    def plot_fun_step(each_step_arg, feed_folder):
        ims = []
        vis_feed = VisFeed()
        for args in each_step_arg["args"]:
            im, info_bin, info_pax = args
            im = IntegratorClass.draw_im(im,
                                         info_bin,
                                         info_pax,
                                         font_scale=0.7)
            ims.append(im)
        # msglist.extend(mlist)
        frame_num = each_step_arg["frame"]
        msglist = each_step_arg["msg"]
        im_feed = vis_feed.draw(ims[0], ims[1], ims[2], frame_num, msglist)

        f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
        cv2.imwrite(str(f_write), cv2.cvtColor(im_feed, cv2.COLOR_RGB2BGR))
        # print(f"save to {f_write}")

    def _extract_info_for_draw_frame(self, out1, out2, out3=None):

        out_args = []  # for multiprocess

        im1, imfile1, _ = out1
        im2, imfile2, _ = out2
        im3 = None
        if out3 is not None:
            im3, imfile3, _ = out3

        frame_num = frame_to_time(imfile1)
        # Cam 09
        info_bin, info_pax, event_bin, event_pax, msglist, logs = self.Info.get_info_from_frame(
            frame_num, "cam09")
        self.full_log.extend(logs)
        if self.plot:
            # im1 = self.Info.draw_im(im1, info_bin, info_pax, font_scale=0.75)
            out_args.append((im1, info_bin, info_pax))

        # print("#draw:1", flush=True)
        # Cam 11
        info_bin, info_pax, event_bin, event_pax, mlist, logs = self.Info.get_info_from_frame(
            frame_num, "cam11")
        msglist.extend(mlist)
        self.full_log.extend(logs)
        if self.plot:
            # im2 = self.Info.draw_im(im2, info_bin, info_pax, font_scale=0.7)
            out_args.append((im2, info_bin, info_pax))

        if out3 is not None:
            frame_num3 = frame_to_time(imfile3)
            info_bin, info_pax, event_bin, event_pax, mlist, logs = self.Info.get_info_from_frame(
                frame_num3, "cam13")
            msglist.extend(mlist)
            self.full_log.extend(logs)
            if self.plot:
                # im3 = self.Info.draw_im(im3,
                #                         info_bin,
                #                         info_pax,
                #                         font_scale=0.7)
                out_args.append((im3, info_bin, info_pax))

        return {
            "args": out_args,
            "frame": frame_num,
            "msg": list(self.Info.msg)
        }

    def finish(self):
        write_file = Path(self.config.rpi_all_results_csv)
        write_file.parent.mkdir(exist_ok=True, parents=True)

        with open(str(write_file), "w") as fp:
            fp.write("\n".join(self.full_log))
