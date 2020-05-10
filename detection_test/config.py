import argparse
import os
import sys
from os.path import expanduser

_HOME = os.environ["HOME"]


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description="Arguments for program")
    parser.add_argument(
        "--root",
        type=str,
        default="~/dataset/ALERT/alert_frames/20191024",
        help="root direcotory of all frames",
    )

    # NOTE: change output directory to save frames and logs
    parser.add_argument(
        "--out_dir",
        "--out",
        default="~/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    parser.add_argument(
        "--bin-ckpt",
        type=str,
        default="~/dataset/ALERT/trained_model/model_cam9_11_13_14.pkl",
    )

    parser.add_argument("--write",
                        action="store_false",
                        help="whether to write frames and info to file")

    parser.add_argument(
        "--size",
        type=str,
        default="640x360",
        help="image size(width x height)",
    )

    parser.add_argument('--run-detector',
                        action='store_true',
                        help='whether to run detector or load pre-run results')

    parser.add_argument(
        "--info",
        default=
        "~/dataset/ALERT/clasp_data/output/run/info_exp1_test_cam09.csv",
        # default=None,
        type=str,
        help="info file to save/load",
    )

    # NOTE: exp name and cameras to use
    parser.add_argument("--file-num", type=str, default="exp1_test")
    parser.add_argument("--cameras", type=str, nargs="*", default=["cam09"])

    parser.add_argument("--start-frame", type=int, default=8600)
    parser.add_argument("--skip-end", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--plot", action="store_true")

    # NOTE: DEBUG only
    parser.add_argument("--load-prev-exit-info",
                        action="store_true",
                        help="whether to load exit info from previous cameras")
    parser.add_argument(
        "--info-prev",
        # default=
        # "~/dataset/ALERT/clasp_data/output/run/info_exp1_test_cam09.csv",
        default=None,
        type=str,
        help="info file to save/load",
    )
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

    # NOTE: change output directory to save frames and logs
    parser.add_argument(
        "--out_dir",
        "--out",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    return parser


parser = get_arg()

if os.uname()[1] == 'lambda-server':  # code is in clasp server
    parser = add_server_specific_arg(parser)

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
    conf.fmt_filename_src = conf.root + "/20191024-training-{cam}{file_num}"
    conf.fmt_filename_out = conf.out_dir + "/run/{file_num}/{cam}"
    conf.fmt_filename_out_feed = conf.out_dir + "/run/feed/{file_num}/{cam}"
    conf.fmt_filename_out_pkl = conf.out_dir + "/out_pkl/{file_num}_{cam}.pkl"
else:
    conf.fmt_filename_src = conf.root + "/{file_num}/{cam}"
    conf.fmt_filename_out = conf.out_dir + "/run/{file_num}/{cam}"
    conf.fmt_filename_out_feed = conf.out_dir + "/run/feed/{file_num}/{cam}"
    conf.fmt_filename_out_pkl = conf.out_dir + "/out_pkl/{file_num}_{cam}.pkl"
