"""
Convert RPI detection file to NU-compatible
"""

import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--file-paths",
                    default=["~/dataset/ALERT/clasp_data/output/run/info.csv"],
                    nargs="*")
parser.add_argument("--out-path", type=str, default="info_for_nu")

args = parser.parse_args()

args.out_path = os.path.expanduser(args.out_path)

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

for file_path in args.file_paths:
    file_path = os.path.expanduser(file_path)

    df = pd.read_csv(
        file_path,
        sep=",",
        header=None,
        names=[
            "file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2",
            "type", "msg"
        ],
        index_col=None,
    )

    print(df.head())

    df_comb = df

    df_comb = df_comb.sort_values('frame')
    df_comb = df_comb[df_comb['type'] == 'loc']

    # up-scale video resolution : 3x
    df_comb["x1"] = 3 * df_comb["x1"]
    df_comb["y1"] = 3 * df_comb["y1"]
    df_comb["x2"] = 3 * df_comb["x2"]
    df_comb["y2"] = 3 * df_comb["y2"]

    write_path = os.path.join(args.out_path, os.path.basename(file_path))
    df_comb.to_csv(write_path, header=None, index=False)