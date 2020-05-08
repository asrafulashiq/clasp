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
        default=_HOME + "/dataset/ALERT/alert_frames/20191024",
        help="root direcotory of all frames",
    )

    # NOTE: change output directory to save frames and logs
    parser.add_argument(
        "--out_dir",
        "--out",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    parser.add_argument(
        "--bin-ckpt",
        type=str,
        default=_HOME + "/dataset/ALERT/trained_model/model.pkl",
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
        "/data/home/islama6/dataset/ALERT/clasp_data/output/run/info_exp1_test_cam09cam11.csv",
        # default=None,
        type=str,
        help="info file to save/load",
    )

    # NOTE: exp name and cameras to use
    parser.add_argument("--file-num", type=str, default="exp1_test")
    parser.add_argument("--cameras", type=str, nargs="*", default=["cam09"])

    parser.add_argument("--start-frame", type=int, default=8000)
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
        # "/data/home/islama6/dataset/ALERT/clasp_data/output/run/info_exp1_test_cam09.csv",
        default=None,
        type=str,
        help="info file to save/load",
    )
    return parser.parse_args()


conf = get_arg()
conf.size = [int(x) for x in conf.size.split("x")]
conf.root = expanduser(conf.root)
conf.out_dir = expanduser(conf.out_dir)
conf.bin_ckpt = expanduser(conf.bin_ckpt)
conf.info = expanduser(conf.info) if conf.info is not None else None