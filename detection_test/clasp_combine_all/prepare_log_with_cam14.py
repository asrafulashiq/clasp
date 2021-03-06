import sys
sys.path.append("../")

import pandas as pd
import os
import numpy as np
from parse import parse
from collections import defaultdict
from tqdm import tqdm
from config import conf
from tools import read_nu_association, read_mu, read_rpi
from rich import print as rprint

# PAX and Bin detection files
if conf.file_num == "exp2":
    bin_file = "./info/info_exp2_cam09cam11cam13.csv"

    pax_file_9 = "./info/cam09exp2_logs_full_may14.txt"
    pax_file_11 = "./info/cam11exp2_logs_full_may14.txt"
    pax_file_13 = "./info/cam13exp2_logs_full_may14.txt"
    pax_file_14 = "./info/cam14exp2_logs_full_may14.txt"

    nu_file_cam9 = "./info/events_training_cam09exp2_102419.csv"
    nu_file_cam11 = "./info/events_training_cam11exp2_102419.csv"
    nu_file_cam13 = "./info/events_training_cam13exp2_102419.csv"
elif conf.file_num == "exp1":
    bin_file = "./info/info_exp1_cam09cam11cam13.csv"

    pax_file_9 = "./info/cam09exp1_logs_full_may14.txt"
    pax_file_11 = "./info/cam11exp1_logs_full_may14.txt"
    pax_file_13 = "./info/cam13exp1_logs_full_may14.txt"

    nu_file_cam9 = "./info/events_test_cam09exp1_102419.csv"
    nu_file_cam11 = "./info/events_test_cam11exp1_102419.csv"
    nu_file_cam13 = "./info/events_test_cam13exp1_102419.csv"

################   DATA PROCSSING #############
# Bin
df_bin = read_rpi(bin_file, scale=3)
# df_bin["frame"] = df_bin[
#     "frame"] + 50  # FIXME: For some reason, there is a 50 frame lag

# PAX
df_pax = read_mu([pax_file_9, pax_file_11, pax_file_13, pax_file_14])

# NU
asso_info, theft_info = {}, {}
asso_info["cam09"], theft_info["cam09"] = read_nu_association(nu_file_cam9)
asso_info["cam11"], theft_info["cam11"] = read_nu_association(nu_file_cam11)
asso_info["cam13"], theft_info["cam13"] = read_nu_association(nu_file_cam13)

# -------------------------------- Create Log -------------------------------- #
full_log = defaultdict(list)

df_comb = pd.concat((df_bin, df_pax), ignore_index=True).sort_values('frame')

for _, row in tqdm(df_comb.iterrows(),
                   total=df_comb.shape[0],
                   desc="Processing : "):
    camera, frame, _id, x1, y1, x2, y2, _type, _class = row[[
        'camera', 'frame', 'id', 'x1', 'y1', 'x2', 'y2', 'type', 'class'
    ]]
    cam = camera[3:5]  # 'cam09' --> '09'

    if _type not in ("loc", "chng", "empty"):
        continue

    if _class == 'dvi':
        type_log = 'DVI'
    else:
        if _class == 'tso':
            type_log = "TSO"
        else:
            type_log = "PAX"

    if _class == 'dvi':
        pax_id = "NA"

        # owner's id is collected from camera 09
        if _id in asso_info["cam09"]:
            ffs = list(asso_info["cam09"][_id])
            for _f in ffs:
                if frame >= _f:
                    pax_id = asso_info["cam09"][_id][_f]
        if _type == "loc":
            # LOC: type: DVI camera-num: 11 frame: 3699 time-offset: 123.3 BB: 1785, 258, 1914, 549
            # ID: B2 PAX-ID: P1 left-behind: false
            if camera in (
                    "cam13", "cam14"
            ) and frame > 9410 and _id == "B27" and conf.file_num == "exp2":
                log_msg = (
                    f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                    +
                    f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} " +
                    "left-behind: true")
            else:
                log_msg = (
                    f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                    +
                    f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} " +
                    "left-behind: false")  # FIXME: left-behind calculation
        elif _type in ("chng", "empty"):
            # XFR: type: FROM camera-num: 13 frame: 4765 time-offset: 158.83
            # BB: 1353, 204, 1590, 462 owner-ID: P2 DVI-ID: B5 theft: FALSE

            owner_id = pax_id

            # FIXME: xfr type to in cam 09 and from in other cameras
            xfr_type = 'TO' if cam == '09' else 'FROM'
            if _type == "empty":
                xfr_type = 'FROM'
            # check potential theft
            _theft = "FALSE"
            if _id in theft_info[camera]:
                ffs = theft_info[camera][_id]
                for _f in ffs:
                    if np.abs(frame - _f) < 100:
                        _theft = "TRUE"
                        pax_id = ffs[_f]
                        xfr_type = 'FROM'
                        del ffs[_f]
                        break

            if pax_id != "NA":
                # get pax BB
                paxes = df_comb[(df_comb['class'] == 'pax')
                                & (df_comb['id'] == pax_id)
                                & (df_comb['camera'] == camera)]
                _ind = (paxes['frame'] - frame).abs().idxmin()
                if np.abs(paxes.loc[_ind]['frame'] - frame) < 30:
                    x1, y1, x2, y2 = paxes.loc[_ind][['x1', 'y1', 'x2', 'y2']]
                    # NOTE: decrease frame number in xfr event
                    log_msg = (
                        f"XFR: type: {xfr_type} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                        +
                        f"BB: {x1}, {y1}, {x2}, {y2} owner-ID: {owner_id} DVI-ID: {_id} theft: {_theft}"
                    )  # REVIEW: 'theft'??

    elif _class in ('pax', 'tso'):
        # LOC: type: PAX camera-num: 13 frame: 4358 time-offset: 145.27 BB: 914, 833, 1190, 1079 ID: P1
        if "TSO" in _id: type_log = "TSO"
        log_msg = (
            f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
            + f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id}")

    full_log[cam].append(log_msg)

for each_cam in full_log:
    fp_writename = f"ata_output/ata_{each_cam}{conf.file_num}.txt"
    with open(fp_writename, 'w') as fp:
        fp.write('\n'.join(full_log[each_cam]))
    rprint(
        f"Write camera [green]{each_cam}[/green] to log file : [italic yellow]{fp_writename}[/italic yellow]"
    )
