import cv2
from tqdm import tqdm
import tools.utils as utils
from config import get_parser, get_conf, add_server_specific_arg
from tools.clasp_logger import ClaspLogger
from manager.main_manager import Manager
import skimage
import os
import numpy as np
from loguru import logger
import pathlib
import torch

import yaml
from colorama import init, Fore
import argparse
from typing import Any, Sequence, Dict, List, Optional, Tuple
import time


class BatchPrecessingMain(object):
    def __init__(self) -> None:
        self.params = self.config_parser()

        self.params.run_detector = True

        self.logger = ClaspLogger()

        self.cameras = [f"cam{int(x):02d}" for x in self.params.cameras]

        self._last_time: Dict[str, List[int]] = {
            cam: [0, -1]
            for cam in self.cameras
        }
        self.manager = Manager(config=self.params,
                               log=self.logger,
                               bin_only=True)

    def on_before_batch_processing(self) -> None:
        self._batch_files_to_process: Dict[str, List] = {
            cam: []
            for cam in self.cameras
        }
        self.out_folder = {}
        for cam in self.cameras:
            self.out_folder[cam] = pathlib.Path(
                self.params.fmt_filename_out.format(
                    file_num=self.params.file_num, cam=cam))
            self.out_folder[cam].mkdir(parents=True, exist_ok=True)

    def process_batch_step(self) -> None:
        if self._get_batch_file_names() is False:
            return

        # load images for all cameras
        image_data = {}
        for cam in self.cameras:
            image_data[cam] = self.load_images_from_files(
                self._batch_files_to_process[cam],
                size=self.params.size,
                file_only=False)

        # process images
        pbar = tqdm(zip(*image_data.values()))
        for _, ret in enumerate(pbar):
            frame_num, i_cnt = 0, 0
            for i_cnt in range(len(ret)):
                im, imfile, frame_num = ret[i_cnt]
                new_im = self.manager.run_detector_image(
                    im, cam=self.cameras[i_cnt], frame_num=frame_num)
                skimage.io.imsave(
                    str(self.out_folder[self.cameras[i_cnt]] / imfile.name),
                    new_im)
            if self.params.write:
                self.manager.write_info()

            if self.cameras[i_cnt] == "cam13":
                frame_num += 50
            pbar.set_description(f"{Fore.CYAN} Processing: {frame_num}")

        if self.params.write:
            self.manager.final_write()

    def on_after_batch_processing(self) -> None:
        pass

    def run(self):
        is_end_of_frames = self._get_flag(flag_name="No_More_Frames")

        while is_end_of_frames == 'FALSE':
            # check whether Frames_Ready_RPI is True
            is_frame_ready = self._get_flag(flag_name="Frames_Ready_RPI")

            if is_frame_ready == 'TRUE':
                self.on_before_batch_processing()
                self.process_batch_step()
            else:
                time.sleep(1)  # pause for 1 sec

            is_end_of_frames = self._get_flag(flag_name="No_More_Frames")

    def _get_batch_file_names(self) -> bool:
        """ return: True denotes there are some files to process """
        flag = False
        for cam, cam_num in zip(self.cameras, self.params.cameras):
            last_time_sec, last_time_ms = self._last_time[cam]

            skip_f = 0
            while skip_f < 5:
                cur_sec, cur_msec = self.increment_time(
                    last_time_sec, last_time_ms)

                # file name
                fname = os.path.join(
                    self.params.root,
                    f"cam{cam_num}_{cur_sec}_{cur_msec:02d}.jpg")
                if not os.path.exists(fname):
                    skip_f += 1
                else:
                    skip_f = 0
                    flag = True
                    self._batch_files_to_process[cam].append(fname)
                last_time_sec, last_time_ms = cur_sec, cur_msec
            self._last_time[cam] = [last_time_sec, last_time_ms]
        return flag

    def _get_flag(self, flag_name):
        value = None
        with open(self.params.flag_file, 'r') as fp:
            data = yaml.full_load(fp)
            try:
                value = data[flag_name]
            except KeyError:
                self.logger.warning(f"No flag {flag_name}!!")
        return value

    @staticmethod
    def load_images_from_files(
            file_list: List[str],
            size=(640, 360),
            file_only=False) -> List[Tuple[np.ndarray, str, int]]:
        """ get images as numpy array from a folder"""
        data = []
        for imfile in file_list:
            imfile = pathlib.Path(imfile)
            if not imfile.exists():
                logger.info(imfile, "does not exist")
                continue
            if file_only:
                image = None
            else:
                image = skimage.io.imread(str(imfile))
                image = cv2.resize(image,
                                   tuple(size),
                                   interpolation=cv2.INTER_LINEAR)
            # get frame number
            frame_num = int(
                round(float('.'.join(imfile.stem.split('_')[-2:])) * 30))
            data.append((image, imfile, frame_num))
        return data

    @staticmethod
    def increment_time(sec, msec, step=1):
        _msec = msec + step
        msec = _msec % 100
        sec += (_msec // 100)
        return sec, msec

    @staticmethod
    def config_parser() -> argparse.Namespace:
        parser = get_parser()
        if os.uname()[1] == 'lambda-server':  # code is in clasp server
            parser = add_server_specific_arg(parser)
        parser = argparse.ArgumentParser(parents=[parser],
                                         conflict_handler='resolve')

        # add parser for video access
        parser.add_argument(
            "--root",
            type=str,
            default="/data/ALERT-SHARE/alert-api-wrapper-data",
            help="root direcotory of all frames",
        )
        parser.add_argument("--file-num", type=str, default="exp2_training")
        parser.add_argument("--cameras",
                            type=str,
                            nargs="*",
                            default=["9", "11", "13"])

        parser.add_argument(
            "--flag_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/FrameSyncFlags.yaml"
        )
        conf = get_conf(parser)
        return conf


if __name__ == "__main__":
    runner = BatchPrecessingMain()
    runner.run()
