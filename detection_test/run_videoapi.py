import torch
torch.backends.cudnn.benchmark = True

from pathlib import Path
import cv2
from tqdm import tqdm
from config import get_parser, get_conf, add_server_specific_arg
from tools.clasp_logger import ClaspLogger
from manager.main_manager import Manager
import skimage
import os
import numpy as np
from loguru import logger
import pathlib
import copy
from colorama import init, Fore
import argparse
from typing import Any, Sequence, Dict, List, Optional, Tuple
import time
from tools.utils_video_api import YAMLCommunicator
from tools.time_calculator import ComplexityAnalysis
from tools.utils_feed import DrawClass
import PIL
from colorama import init

init(autoreset=True)

complexity_analyzer = ComplexityAnalysis()


class BatchPrecessingMain(object):
    def __init__(self) -> None:

        self.params = self.config_parser()
        self.params.run_detector = True
        self.logger = ClaspLogger()

        cam = copy.deepcopy(self.params.cameras)
        self.params.cameras = [f"cam{int(x):02d}" for x in self.params.cameras]
        self.cameras = self.params.cameras
        self.cameras_short = cam

        complexity_analyzer.frames_per_batch = int(
            self.params.duration * self.params.fps * len(self.cameras))

        self._last_frame: Dict[str, int] = {
            cam: self.params.start_frame - 1
            for cam in self.cameras
        }
        self.manager = Manager(config=self.params,
                               log=self.logger,
                               bin_only=True,
                               analyzer=complexity_analyzer)
        self.yaml_communicator = YAMLCommunicator(
            self.params.flag_file,
            rpi_flagfile=self.params.rpi_flagfile,
            neu_flagfile=self.params.neu_flagfile,
            mu_flagfile=self.params.neu_flagfile)
        self.drawing_obj = DrawClass(conf=self.params, plot=True)

        # output folder name for each camera
        self.out_folder = {}
        for cam in self.cameras:
            self.out_folder[cam] = pathlib.Path(
                self.params.fmt_filename_out.format(
                    file_num=self.params.file_num, cam=cam))
            self.out_folder[cam].mkdir(parents=True, exist_ok=True)

        print(self.params)

    def process_batch_step(self) -> None:
        self.yaml_communicator.set_bin_processed(value='FALSE')

        # Read batch files
        with complexity_analyzer("READ"):
            batch_files_to_process, flag = self._get_batch_file_names(
                load_image=True)
        if flag is False:
            return

        # fresh dict for storing the batch info only
        self.manager.init_data_dict()

        with complexity_analyzer("RPI_only"):
            # calculate detection
            with complexity_analyzer("DET"):
                for cam, value in batch_files_to_process.items():
                    imlist = [v[0] for v in value]
                    result_det_list = self.manager.pre_calculate_detector(
                        imlist,
                        return_im=False,
                        max_batch=self.params.max_batch_detector)
                    for iret, det in enumerate(result_det_list):
                        value[iret].append(det)

            cam_frame_num, i_cnt = None, None
            batch_frames = []

            # load images for all cameras
            pbar = tqdm(zip(*batch_files_to_process.values()),
                        total=len(batch_files_to_process[self.cameras[0]]),
                        position=5)
            with complexity_analyzer("BIN_PROCESS"):
                for _, ret in enumerate(pbar):
                    _rets_cam = []
                    for i_cnt in range(len(ret)):
                        # complexity_analyzer.start("READ")
                        # im, imfile, frame_num = self.load_images_from_files(
                        #     [ret[i_cnt]], size=self.params.size, file_only=False)[0]
                        # complexity_analyzer.pause("READ")

                        im, imfile, frame_num, det = ret[i_cnt]
                        with complexity_analyzer("TRACK", False):
                            new_im = self.manager.run_tracking_per_frame(
                                im,
                                det,
                                cam=self.cameras[i_cnt],
                                frame_num=frame_num)

                        if self.params.save_im == "true":
                            skimage.io.imsave(
                                str(self.out_folder[self.cameras[i_cnt]] /
                                    Path(imfile).name), new_im)
                        cam_frame_num = frame_num
                        _rets_cam.append((im, imfile, frame_num))

                    batch_frames.append(_rets_cam)
                    if self.params.write:
                        self.manager.load_info()

                    # if self.cameras[i_cnt] == "cam13":
                    #     frame_num += 50
                    pbar.set_description(
                        f"{Fore.CYAN} Processing: {cam_frame_num}")

                if self.params.write:
                    df_batch = self.manager.get_batch_info()
                    write_path = self.params.rpi_result_file
                    # write output files for NU
                    df_batch['timeoffset'] = df_batch['frame'].apply(
                        lambda frame: '{:.2f}'.format(frame / 30.0))
                    df_batch.to_csv(write_path, index=False, header=True)
                pbar.close()

        # tell NU that bin is processed
        self.yaml_communicator.set_bin_processed(value='TRUE')

        # TODO: create combined log and feed
        if self.params.debug:
            self.logger.debug("DEBUG: mock create combined output")
            if self.params.create_feed:
                self.on_after_batch_processing(batch_frames)
        else:
            while True:
                ret = self.yaml_communicator.is_association_ready()
                if ret:
                    self.on_after_batch_processing(batch_frames)
                    break

        # Set batch processed flag
        self.yaml_communicator.set_batch_processed()

        while True:
            if self.params.debug:
                # set people processed
                self.yaml_communicator._set_flag('People_Processed', 'TRUE',
                                                 self.params.mu_flagfile)
                # set association ready
                self.yaml_communicator._set_flag('Association_Ready', 'TRUE',
                                                 self.params.neu_flagfile)
                if (self.yaml_communicator.check_next_batch_ready()):
                    self.yaml_communicator.set_batch_processed(value='FALSE')
                    self.logger.info("Batch processed set to FALSE")
                    break
            else:
                if (self.yaml_communicator.check_next_batch_ready()
                        and not self.yaml_communicator.
                        check_people_processed_ready()
                        and not self.yaml_communicator.is_association_ready()):
                    self.yaml_communicator.set_batch_processed(value='FALSE')
                    self.logger.info("Batch processed set to FALSE")
                    break

        # tell NU that bin is not processed
        self.yaml_communicator.set_bin_processed(value='FALSE')

    def on_after_batch_processing(self, batch_frames) -> None:
        with complexity_analyzer("DRAW"):
            # self.drawing_obj.run_process(batch_frames,
            #                              self.params.rpi_result_file,
            #                              self.params.mu_result_file,
            #                              self.params.neu_result_file)
            self.drawing_obj.draw_batch(batch_frames,
                                        self.params.rpi_result_file,
                                        self.params.mu_result_file,
                                        self.params.neu_result_file)

    def run(self):
        with complexity_analyzer("INIT"):
            counter = 0
            pbar = tqdm(position=1)
            while self.yaml_communicator.is_end_of_frames() is False:
                if self.yaml_communicator.is_batch_ready():
                    with complexity_analyzer("BATCH"):
                        self.process_batch_step()

                    complexity_analyzer.current_memory_usage()
                    complexity_analyzer.get_time_info()

                else:
                    time.sleep(0.1)  # pause for 0.1 sec
                pbar.set_description(Fore.YELLOW + f"Loop {counter}")
                pbar.update()
                counter += 1
            pbar.close()
        complexity_analyzer.final_info()

    def _get_batch_file_names(self, load_image=False) -> bool:
        """ return: True denotes there are some files to process """
        batch_files_to_process: Dict[str, List] = {
            cam: []
            for cam in self.cameras
        }

        flag = False
        for cam, cam_num in zip(self.cameras, self.cameras_short):
            last_frame = self._last_frame[cam]

            skip_f = 0
            while skip_f < 5:
                cur_frame = last_frame + 1

                # file name
                fname = os.path.join(self.params.root,
                                     f"cam{cam_num}_{cur_frame:06d}.jpg")
                if not os.path.exists(fname):
                    skip_f += 1
                else:
                    skip_f = 0
                    flag = True
                    if load_image:
                        frame_num = int(Path(fname).stem.split('_')[-1])
                        image = self.read_image(fname, size=self.params.size)
                        batch_files_to_process[cam].append(
                            [image, str(fname), frame_num])
                    else:
                        batch_files_to_process[cam].append(fname)
                last_frame = cur_frame
            self._last_frame[cam] = last_frame
        return batch_files_to_process, flag

    def load_images_from_files(self,
                               file_list: List[str],
                               size=(640, 360),
                               file_only=False
                               ) -> List[Tuple[np.ndarray, str, int]]:
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
                image = self.read_image(imfile)
            # get frame number
            frame_num = int(imfile.stem.split('_')[-1])
            data.append((image, imfile, frame_num))
        return data

    @staticmethod
    def read_image(imfile, size=(640, 360)):
        image = cv2.cvtColor(cv2.imread(str(imfile)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
        return image

    def config_parser(self) -> argparse.Namespace:
        parser = get_parser()
        if os.uname()[1] == 'lambda-server':  # code is in clasp server
            parser = add_server_specific_arg(parser)
        parser = argparse.ArgumentParser(parents=[parser],
                                         conflict_handler='resolve')

        # parser.add_argument("--show_info")

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
                            default=["9", "11"])

        parser.add_argument(
            "--flag_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_Wrapper.yaml"
        )
        parser.add_argument("--max_files_in_batch", type=int, default=30)
        parser.add_argument("--debug", "-d", action="store_true")
        parser.add_argument(
            "--batch_out_folder",
            type=str,
            default="/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/")
        parser.add_argument(
            "--rpi_flagfile",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_RPI.yaml"
        )

        parser.add_argument(
            "--mu_flagfile",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_MU.yaml"
        )

        parser.add_argument(
            "--neu_result_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/NEU/log_batch_association.csv"
        )

        parser.add_argument(
            "--mu_result_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/mu/log_batch_mu_current.csv"
        )

        parser.add_argument(
            "--rpi_result_file",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/rpi_result.csv"
        )

        parser.add_argument(
            "--neu_flagfile",
            type=str,
            default=
            "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_NEU.yaml"
        )

        parser.add_argument("--duration", type=float, default=4)
        parser.add_argument("--fps", type=int, default=10)

        parser.add_argument("--save_im", type=str, default="true")

        if parser.parse_known_args()[0].debug:
            parser.add_argument(
                "--flag_file",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_Wrapper.yaml")
            parser.add_argument("--max_files_in_batch", type=int, default=30)
            parser.add_argument("--debug", action="store_true")
            parser.add_argument(
                "--batch_out_folder",
                type=str,
                default=
                "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/")
            parser.add_argument(
                "--rpi_flagfile",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_RPI.yaml")

            parser.add_argument(
                "--mu_flagfile",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_MU.yaml")

            parser.add_argument(
                "--neu_result_file",
                type=str,
                default="/home/rpi/data/wrapper_log/log_batch_association.csv")

            parser.add_argument(
                "--mu_result_file",
                type=str,
                default="/home/rpi/data/wrapper_log/log_batch_mu_current.csv")

            parser.add_argument(
                "--rpi_result_file",
                type=str,
                default="/home/rpi/data/wrapper_log/rpi_result.csv")

            parser.add_argument(
                "--neu_flagfile",
                type=str,
                default="/home/rpi/data/wrapper_log/Flags_NEU.yaml")

            parser.add_argument(
                "--root",
                type=str,
                default="/home/rpi/data/wrapper_data/",
                help="root direcotory of all frames",
            )

        # NOTE where to start
        parser.add_argument("--start_frame", type=int, default=0)

        conf = get_conf(parser)
        return conf


if __name__ == "__main__":
    runner = BatchPrecessingMain()
    runner.run()
