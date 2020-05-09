import pandas as pd
import os
import sys
import numpy as np
from collections import defaultdict
from parse import parse


def read_rpi(filename, scale=3):
    """ Read dataframe from rpi logfile """
    df = pd.read_csv(
        str(filename),
        sep=",",
        header=None,
        names=[
            "file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2",
            "type", "msg"
        ],
        index_col=None,
    )
    df = df.sort_values("frame")
    df['class'] = 'dvi'

    # resize to original image size
    df["x1"] = 3 * df["x1"]
    df["y1"] = 3 * df["y1"]
    df["x2"] = 3 * df["x2"]
    df["y2"] = 3 * df["y2"]
    return df


def read_mu(filename):
    """Convert MU log files to same format 
    
    Example
        one row from MU csv: `2214,TSO1,611,907,828,1080,cam09exp2,0,LOC`
        note that image size in original size (1920 x 1080)
    """

    if filename is None or not filename:
        return pd.DataFrame()

    if isinstance(filename, list):
        list_df = [read_mu(single_file) for single_file in filename]
        df = pd.concat(list_df, ignore_index=True)
        return df
    else:
        header = [
            "frame", "id", "x1", "y1", "x2", "y2", "camera", "TU", "type"
        ]
        df = pd.read_csv(str(filename),
                         sep=",",
                         header=None,
                         names=header,
                         index_col=None)

        # NOTE: 'camera' in 'cam09exp2' format, convert to 'cam09'
        df["camera"] = df["camera"].apply(lambda x: x[:5])
        df["type"] = df["type"].str.lower()
        df['class'] = 'pax'
        return df


def read_nu_association(nu_file1):
    """Convert NU association info  

    Example:
        `6124,[]`, `6154,['P5-B14']`,
        `11796,"['P17-B36', 'P 16 is possibly stealing sth from -B36', 'P17-B42']"`

    Returns:
        [dict]: key is `bin_id`, and value is a dict
                        where key is `frame_number` and value is associated `pax_id`
    """
    if nu_file1 is None:
        return {}, {}

    asso_info = defaultdict(dict)
    theft_info = defaultdict(dict)

    df = pd.read_csv(str(nu_file1), header=None, names=["frame", "des"])

    for _, row in df.iterrows():
        frame = row["frame"]
        des = row["des"]
        des = parse("[{}]", des)
        if des is None:
            continue
        des = des[0]
        for each_split in des.split(","):
            pp = parse("'P{}-B{}'", each_split)
            if pp is not None:
                pax_id, bin_id = pp[0], int(pp[1])
                if "stealing" in pax_id or 'stoling' in pax_id:
                    # 'P 3 is possibly stoling sth from -B6'
                    _pax, *_ = parse("{:d}{}", pax_id)
                    theft_info[bin_id][frame] = 'P' + str(_pax)
                else:
                    asso_info[bin_id][frame] = 'P' + str(pax_id)
    return asso_info, theft_info