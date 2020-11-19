import argparse
import os
import sys
from os.path import expanduser

if os.uname()[1] == 'lambda-server':
    _HOME = "/home/rpi"
else:
    _HOME = os.environ["HOME"]


def get_parser():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description="Arguments for program")
    parser.add_argument(
        "--root",
        type=str,
        default=_HOME + "/dataset/ALERT/alert_frames/20191024",
        help="root direcotory of all frames",
    )

    # NOTE: change output directory to save frames and logs
    parser.add_argument(
        "--out_dir",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    parser.add_argument(
        "--bin_ckpt",
        type=str,
        default=_HOME + "/dataset/ALERT/trained_model/weight_1280x720_aug.pkl",
    )

    parser.add_argument("--write",
                        action="store_false",
                        help="whether to write frames and info to file")

    parser.add_argument(
        "--size",
        type=str,
        # default="640x360",
        default="1280x720",
        help="image size(width x height)",
    )

    parser.add_argument("--fp16",
                        action="store_true",
                        help="whether to activate 16 bit precision")

    # whether to also run rcnn detector to detect item
    parser.add_argument('--run_detector',
                        action='store_true',
                        help='whether to run detector or load pre-run results')

    parser.add_argument("--max_batch_detector", type=int, default=10)

    parser.add_argument(
        "--info",
        # default=
        # "/home/rpi/dataset/ALERT/clasp_data/output/run/info_exp2_train_cam09cam11cam13.csv",
        default=None,
        type=str,
        help="info file to save/load",
    )

    # NOTE: exp name and cameras to use
    parser.add_argument("--file_num", type=str, default="exp2_train")
    parser.add_argument("--cameras",
                        type=str,
                        nargs="*",
                        default=["cam09", "cam11"])

    parser.add_argument("--start_frame", type=int, default=2500)
    parser.add_argument("--skip_end", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--delta", type=int, default=3)
    parser.add_argument("--plot", action="store_true")

    # NOTE: DEBUG only
    parser.add_argument("--load_prev_exit_info",
                        action="store_true",
                        help="whether to load exit info from previous cameras")
    parser.add_argument(
        "--info_prev",
        # default=
        # "/home/rpi/dataset/ALERT/clasp_data/output/run/info_exp2_train_cam09cam11cam13.csv",
        default=None,
        type=str,
        help="info file to save/load",
    )

    parser.add_argument("--out_suffix", type=str, default="")
    parser.add_argument("--create_feed", "-f", action="store_true")

    parser.add_argument("--spatial_scale_mul", type=float, default=2)
    parser.add_argument("--temporal_scale_mul", type=float, default=3)

    return parser


def add_server_specific_arg(parent_parser):
    """ Handle different data path for different server """
    parser = argparse.ArgumentParser(parents=[parent_parser],
                                     conflict_handler='resolve')

    parser.add_argument(
        "--root",
        type=str,
        default="/data/CLASP-DATA/CLASP2-data/20191024/frames",
        help="root direcotory of all frames",
    )

    parser.add_argument("--mode", type=str, default="training")
    parser.add_argument("--file-num", type=str, default="exp2")

    # NOTE: change output directory to save frames and logs
    parser.add_argument(
        "--out_dir",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    return parser


def get_conf(parser):
    conf = parser.parse_args()
    conf.size = [int(x) for x in conf.size.split("x")]
    conf.root = expanduser(conf.root)
    conf.out_dir = expanduser(conf.out_dir)
    conf.bin_ckpt = expanduser(conf.bin_ckpt)
    conf.info = expanduser(conf.info) if conf.info is not None else None
    conf.info_prev = expanduser(
        conf.info_prev) if conf.info_prev is not None else None

    conf.fmt = "{:06d}.jpg"  # frame name format

    if os.uname()[1] == 'lambda-server':  # code is in clasp server
        conf.fmt = "frame{:05d}.jpg"
        conf.fmt_filename_src = conf.root + "/20191024-" + conf.mode + "-{cam}{file_num}"
    else:
        conf.fmt_filename_src = conf.root + "/{file_num}/{cam}"

    conf.fmt_filename_out = conf.out_dir + "/run/{file_num}" + conf.out_suffix + "/{cam}"
    conf.fmt_filename_out_detection = conf.out_dir + "/out_detection/{file_num}/{cam}"
    conf.fmt_filename_out_feed = conf.out_dir + "/run/feed/{file_num}"
    conf.fmt_filename_out_pkl = conf.out_dir + "/out_pkl/{file_num}_{cam}.pkl"
    return conf