import sys
sys.path.append("../")

import pandas as pd
import os
import numpy as np
from parse import parse
from collections import defaultdict
from tqdm import tqdm
from tools import read_nu_association, read_mu, read_rpi
from rich import print as rprint

# PAX and Bin detection files
bin_file = "./info/info.csv"
pax_file_9 = "./info/cam09exp2_logs_fullv1.txt"
pax_file_11 = "./info/cam11exp2_logs_fullv1.txt"
pax_file_13 = "./info/cam13exp2_logs_fullv1.txt"

nu_file_cam9 = "./info/cam_09_exp2_associated_events.csv"
nu_file_cam11 = "./info/cam_11_exp2_associated_events.csv"

################   DATA PROCSSING #############
# Bin
df_bin = read_rpi(bin_file, scale=3)

# PAX
df_pax = read_mu([pax_file_9, pax_file_11, pax_file_13])

# NU
asso_info, theft_info = read_nu_association(nu_file_cam9)

# -------------------------------- Create Log -------------------------------- #
full_log = defaultdict(list)

df_comb = pd.concat((df_bin, df_pax), ignore_index=True).sort_values('frame')

for _, row in tqdm(df_comb.iterrows(),
                   total=df_comb.shape[0],
                   desc="Processing : "):
    cam, frame, _id, x1, y1, x2, y2, _type, _class = row[[
        'camera', 'frame', 'id', 'x1', 'y1', 'x2', 'y2', 'type', 'class'
    ]]
    cam = cam[3:5]  # 'cam09' --> '09'

    if _type not in ("loc", "chng"):
        continue

    if _class == 'dvi':
        type_log = 'DVI'
    else:
        if "TSO" in _id:
            type_log = "TSO"
        else:
            type_log = "PAX"

    if _class == 'dvi':
        pax_id = "NA"
        if _id in asso_info:
            ffs = list(asso_info[_id])
            for _f in ffs:
                if frame >= _f:
                    pax_id = asso_info[_id][_f]
        if _type == "loc":
            # LOC: type: DVI camera-num: 11 frame: 3699 time-offset: 123.3 BB: 1785, 258, 1914, 549
            # ID: B2 PAX-ID: P1 left-behind: false
            log_msg = (
                f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                + f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} " +
                "left-behind: false")  # FIXME: left-behind calculation
        elif _type == "chng":
            # XFR: type: FROM camera-num: 13 frame: 4765 time-offset: 158.83
            # BB: 1353, 204, 1590, 462 owner-ID: P2 DVI-ID: B5 theft: FALSE

            # FIXME: xfr type to in cam 09 and from in other cameras
            xfr_type = 'TO' if cam == '09' else 'FROM'

            # check potential theft
            _theft = "FALSE"
            if _id in theft_info:
                ffs = theft_info[_id]
                for _f in ffs:
                    if np.abs(frame - _f) < 50:
                        _theft = "TRUE"
                        pax_id = ffs[_f]
                        xfr_type = 'FROM'
                        del ffs[_f]

            if pax_id != "NA":
                log_msg = (
                    f"XFR: type: {xfr_type} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                    +
                    f"BB: {x1}, {y1}, {x2}, {y2} owner-ID: {pax_id} DVI-ID: {_id} theft: {_theft}"
                )  # REVIEW: 'theft'??

    elif _class == 'pax':
        # LOC: type: PAX camera-num: 13 frame: 4358 time-offset: 145.27 BB: 914, 833, 1190, 1079 ID: P1
        log_msg = (
            f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
            + f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id}")

    full_log[cam].append(log_msg)

for each_cam in full_log:
    fp_writename = f"ata_output/ata_{each_cam}.txt"
    with open(fp_writename, 'w') as fp:
        fp.write('\n'.join(full_log[each_cam]))
    rprint(
        f"Write camera [green]{each_cam}[/green] to log file : [italic yellow]{fp_writename}[/italic yellow]"
    )
