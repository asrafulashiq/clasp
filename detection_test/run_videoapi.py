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
import copy
import yaml
from colorama import init, Fore
import argparse
from typing import Any, Sequence, Dict, List, Optional, Tuple
import time
from tools.utils_video_api import YAMLCommunicator


class BatchPrecessingMain(object):
    def __init__(self) -> None:
        self.params = self.config_parser()
        self.params.run_detector = True
        self.logger = ClaspLogger()

        cam = copy.deepcopy(self.params.cameras)
        self.params.cameras = [f"cam{int(x):02d}" for x in self.params.cameras]
        self.cameras = self.params.cameras
        self.cameras_short = cam

        self._last_time: Dict[str, List[int]] = {
            cam: [self.params.start_frame, -1]
            for cam in self.cameras
        }
        self.manager = Manager(config=self.params,
                               log=self.logger,
                               bin_only=True)
        self.yaml_communicator = YAMLCommunicator(
            self.params.flag_file, rpi_flagfile=self.params.rpi_flagfile)

        # output folder name for each camera
        self.out_folder = {}
        for cam in self.cameras:
            self.out_folder[cam] = pathlib.Path(
                self.params.fmt_filename_out.format(
                    file_num=self.params.file_num, cam=cam))
            self.out_folder[cam].mkdir(parents=True, exist_ok=True)

    def process_batch_step(self) -> None:
        self.yaml_communicator.set_bin_processed(value='FALSE')

        # Read batch files
        batch_files_to_process, flag = self._get_batch_file_names()
        if flag is False:
            return
        self.yaml_communicator.set_batch_read()

        # fresh dict for storing the batch info only
        self.manager.init_data_dict()

        # load images for all cameras
        pbar = tqdm(zip(*batch_files_to_process.values()),
                    total=len(batch_files_to_process[self.cameras[0]]),
                    position=5)
        cam_frame_num, i_cnt = None, None
        for _, ret in enumerate(pbar):
            for i_cnt in range(len(ret)):
                im, imfile, frame_num = self.load_images_from_files(
                    [ret[i_cnt]], size=self.params.size, file_only=False)[0]
                new_im = self.manager.run_detector_image(
                    im, cam=self.cameras[i_cnt], frame_num=frame_num)
                skimage.io.imsave(
                    str(self.out_folder[self.cameras[i_cnt]] / imfile.name),
                    new_im)
                cam_frame_num = frame_num
            if self.params.write:
                self.manager.load_info()

            # if self.cameras[i_cnt] == "cam13":
            #     frame_num += 50
            pbar.set_description(f"{Fore.CYAN} Processing: {cam_frame_num}")

            if self.params.write:
                df_batch = self.manager.get_batch_info()
                write_path = os.path.join(self.params.batch_out_folder,
                                          "bin_log_batch.csv")
                df_batch.to_csv(write_path, index=False)

        pbar.close()

        # tell NU that bin is processed
        self.yaml_communicator.set_bin_processed(value='TRUE')

        # TODO: write output files for NU
        if self.params.debug:
            self.logger.debug("DEBUG: mock writted outfile files for NU")
        else:
            raise NotImplementedError()

        # TODO: create combined log and feed
        if self.params.debug:
            self.logger.debug("DEBUG: mock create combined output")
        else:
            raise NotImplementedError()

        # Set batch processed flag
        self.yaml_communicator.set_batch_processed()
        # tell NU that bin is not processed
        self.yaml_communicator.set_bin_processed(value='FALSE')

    def on_after_batch_processing(self) -> None:
        pass

    def run(self):
        counter = 0
        pbar = tqdm(position=1)
        while self.yaml_communicator.is_end_of_frames() is False:
            if self.yaml_communicator.is_batch_ready():
                self.process_batch_step()
            else:
                time.sleep(1)  # pause for 1 sec
            pbar.set_description(Fore.RED + f"Loop {counter}")
            pbar.update()
            counter += 1
        pbar.close()

    def _get_batch_file_names(self) -> bool:
        """ return: True denotes there are some files to process """
        batch_files_to_process: Dict[str,
                                     List] = {cam: []
                                              for cam in self.cameras}

        flag = False
        for cam, cam_num in zip(self.cameras, self.cameras_short):
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
                    batch_files_to_process[cam].append(fname)
                last_time_sec, last_time_ms = cur_sec, cur_msec
            self._last_time[cam] = [last_time_sec, last_time_ms]
        return batch_files_to_process, flag

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

    def config_parser(self) -> argparse.Namespace:
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
        parser.add_argument("--file-num", type=str, default="exp2training")
        parser.add_argument("--cameras",
                            type=str,
                            nargs="*",
                            default=["9", "11", "13"])

        parser.add_argument(
            "--flag_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_Wrapper.yaml"
        )
        parser.add_argument("--max_files_in_batch", type=int, default=30)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--batch_out_folder",
            type=str,
            default="/data/ALERT-SHARE/alert-api-wrapper-data/rpi")
        parser.add_argument(
            "--rpi_flagfile",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_RPI.yaml"
        )

        if parser.parse_known_args()[0].debug:
            parser.add_argument(
                "--flag_file",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_Wrapper.yaml")
            parser.add_argument(
                "--root",
                type=str,
                default="/home/rpi/data/wrapper_data/",
                help="root direcotory of all frames",
            )

            parser.add_argument(
                "--rpi_flagfile",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_RPI.yaml")

        parser.add_argument("--start_frame", type=int, default=0)

        conf = get_conf(parser)
        return conf


if __name__ == "__main__":
    runner = BatchPrecessingMain()
    runner.run()
