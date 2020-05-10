import pandas as pd
import os
import sys
import numpy as np
from collections import defaultdict
from parse import parse


def read_rpi(filename, scale=None):
    """ Read dataframe from rpi logfile """
    df = pd.read_csv(
        str(filename),
        sep=",",
        header=None,
        names=[
            "file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2",
            "type", "msg"
        ],
        dtype={
            'file': str,
            'msg': str,
            'type': str,
            'camera': str,
            'class': str,
            'frame': int,
            'id': str,
            'x1': int,
            'y1': int,
            'x2': int,
            'y2': int
        },
        index_col=None,
    )

    df['class'] = 'dvi'
    df['id'] = 'B' + df['id'].astype(str)
    # resize to original image size
    if scale is not None:
        df["x1"] = scale * df["x1"]
        df["y1"] = scale * df["y1"]
        df["x2"] = scale * df["x2"]
        df["y2"] = scale * df["y2"]
    return df


def read_mu(filename, scale=None):
    """Convert MU log files to same format 
    
    Example
        one row from MU csv: `LOC: type: PAX camera-num: 9 frame: 1628 time-offset: 54 BB: 330,874,581,1075 ID: 27 
            PAX-ID: TSO1 first-used: true partial-complete: description:`
        note that image size in original size (1920 x 1080)
    """

    if filename is None or not filename:
        return pd.DataFrame()

    if isinstance(filename, list):
        list_df = [read_mu(single_file, scale) for single_file in filename]
        df = pd.concat(list_df, ignore_index=True)
        return df
    else:
        tmp_dict = defaultdict(list)
        with open(str(filename), 'r') as fp:
            for line in fp.readlines():
                line = line.strip()
                out = parse(
                    "LOC: type: {class} camera-num: {camera} frame: {frame} time-offset: {} BB: {x1},{y1},{x2},{y2} ID: {} PAX-ID: {pax_id} first-used:{}",
                    line)
                if out is not None:
                    tmp_dict['camera'].append(out['camera'])
                    tmp_dict['frame'].append(out['frame'])
                    tmp_dict['class'].append(out['class'].lower())
                    tmp_dict['id'].append(out['pax_id'])
                    tmp_dict['x1'].append(out['x1'])
                    tmp_dict['y1'].append(out['y1'])
                    tmp_dict['x2'].append(out['x2'])
                    tmp_dict['y2'].append(out['y2'])
                else:
                    pass

        df = pd.DataFrame(data=tmp_dict)
        df = df.astype(
            dtype={
                'camera': str,
                'class': str,
                'frame': int,
                'id': str,
                'x1': int,
                'y1': int,
                'x2': int,
                'y2': int
            })

        df['camera'] = df['camera'].apply(lambda x: f"cam{int(x):02d}")
        # df["type"] = df["type"].str.lower()
        df['type'] = 'loc'
        # df['class'] = 'pax'

        # df[['frame', 'id', 'x1', 'y1', 'x2',
        #     'y2']] = df[['frame', 'id', 'x1', 'y1', 'x2',
        #                  'y2']].astype(np.int64)

        if scale is not None:
            df["x1"] = df["x1"] * scale
            df["y1"] = df["y1"] * scale
            df["x2"] = df["x2"] * scale
            df["y2"] = df["y2"] * scale
        return df


def read_nu_association(nu_file1):
    """Convert NU association info  

    Example:
        5242,"['|P5| left hand in |Bin 14|', '|P5| is the owner of |Bin 14|']
        5628,"['|P5| right hand in |Bin 18|', '|P5| is suspicious with |Bin 18|']"
        5215,['|P5| left hand in |Bin 14|']

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
            each_split = each_split.strip()
            pp = parse("'|P{pax_id:d}|{event_str}|Bin {bin_id:d}|'",
                       each_split)
            if pp is not None:
                bin_id, pax_id = 'B' + str(pp['bin_id']), 'P' + str(
                    pp['pax_id'])
                if "owner of" in pp['event_str']:
                    # association
                    asso_info[bin_id][frame] = pax_id
                elif "hand in" in pp['event_str']:
                    pass
                elif "suspicious" in pp['event_str']:
                    theft_info[bin_id][frame] = pax_id
    return asso_info, theft_info


# def read_mu(filename, scale=None):
#     """Convert MU log files to same format

#     Example
#         one row from MU csv: `2214,TSO1,611,907,828,1080,cam09exp2,0,LOC`
#         note that image size in original size (1920 x 1080)
#     """

#     if filename is None or not filename:
#         return pd.DataFrame()

#     if isinstance(filename, list):
#         list_df = [read_mu(single_file, scale) for single_file in filename]
#         df = pd.concat(list_df, ignore_index=True)
#         return df
#     else:
#         header = [
#             "frame", "id", "x1", "y1", "x2", "y2", "camera", "TU", "type"
#         ]
#         df = pd.read_csv(str(filename),
#                          sep=",",
#                          header=None,
#                          names=header,
#                          index_col=None)

#         # NOTE: 'camera' in 'cam09exp2' format, convert to 'cam09'
#         df["camera"] = df["camera"].apply(lambda x: x[:5])
#         df["type"] = df["type"].str.lower()
#         df['class'] = 'pax'

#         if scale is not None:
#             df["x1"] = df["x1"] * scale
#             df["y1"] = df["y1"] * scale
#             df["x2"] = df["x2"] * scale
#             df["y2"] = df["y2"] * scale

#         return df