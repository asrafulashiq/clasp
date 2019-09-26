import argparse
import os
import sys


_HOME = os.environ["HOME"]


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description="Arguments for program")
    parser.add_argument(
        "--root",
        type=str,
        default=_HOME + "/dataset/ALERT/alert_frames",
        help="root direcotory of all frames",
    )
    parser.add_argument(
        "--out_dir",
        "--out",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    parser.add_argument(
        "--bin-ckpt",
        type=str,
        default=_HOME + "/dataset/clasp/trained_model/model.pkl",
    )

    parser.add_argument(
        "--pax-ckpt",
        type=str,
        default=_HOME + "/dataset/clasp/trained_model/model.pkl",
    )

    parser.add_argument("--write", action="store_false")

    parser.add_argument(
        "--size",
        type=str,
        default="640x360",
        help="image size(width x height)",
    )

    parser.add_argument(
        "--info",
        # default="/home/ash/Desktop/hdd/ALERT/clasp_data/output/run/info.csv",
        default=None,
        type=str,
        help="info file to load",
    )

    parser.add_argument("--skip-init", type=int, default=5961)
    parser.add_argument("--skip-end", type=int, default=1000)
    parser.add_argument("--delta", type=int, default=2)

    return parser.parse_args()


conf = get_arg()
conf.size = [int(x) for x in conf.size.split("x")]

