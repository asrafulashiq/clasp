import pandas as pd
import os
import numpy as np
from parse import parse
from collections import defaultdict
from tqdm import tqdm


def get_association_info(nu_file1, nu_file2):
    asso_info = defaultdict(dict)
    df_tmp = pd.read_csv(nu_file1, header=None, names=["frame", "des"])

    for _, row in df_tmp.iterrows():
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
                if "stealing" in pax_id:
                    pass
                else:
                    asso_info[bin_id][frame] = pax_id
    return asso_info


# PAX and Bin detection files
bin_file = "./info/info_offset.csv"
pax_file_9 = "./info/cam09exp2_logs_fullv1.txt"
pax_file_11 = "./info/cam11exp2_logs_fullv1.txt"
pax_file_13 = "./info/cam13exp2_logs_fullv1.txt"
nu_file_cam9 = "./info/cam_09_exp2_associated_events.csv"
nu_file_cam11 = "./info/cam_11_exp2_associated_events.csv"

################   DATA PROCSSING #############
# Bin
df = pd.read_csv(
    bin_file,
    sep=",",
    header=None,
    names=[
        "file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2",
        "type", "msg"
    ],
    index_col=None,
)

df_new = df[df["type"] == "loc"].copy()
df_new["frame"] = df["frame"] + 1
df_comb = pd.concat((df, df_new))
df_comb = df_comb.sort_values("frame")
df_comb["x1"] = 3 * df_comb["x1"]
df_comb["y1"] = 3 * df_comb["y1"]
df_comb["x2"] = 3 * df_comb["x2"]
df_comb["y2"] = 3 * df_comb["y2"]
df_bin = df_comb

# PAX
pax_names = ["frame", "id", "x1", "y1", "x2", "y2", "camera", "TU", "type"]
df_pax_9 = pd.read_csv(str(pax_file_9),
                       sep=",",
                       header=None,
                       names=pax_names,
                       index_col=None)

df_pax_11 = pd.read_csv(str(pax_file_11),
                        sep=",",
                        header=None,
                        names=pax_names,
                        index_col=None)

df_pax_13 = pd.read_csv(str(pax_file_13),
                        sep=",",
                        header=None,
                        names=pax_names,
                        index_col=None)

df_pax = pd.concat((df_pax_9, df_pax_11, df_pax_13))

# NU
asso_info = get_association_info(nu_file_cam9, nu_file_cam11)

########################## Create Log ###########################
full_logs_9 = []
full_logs_11 = []
full_logs_13 = []

bin_ids = []  # for determining first used
for _, row in tqdm(df_bin.iterrows()):
    cam, frame, _id, x1, y1, x2, y2, _type = (
        row["camera"],
        row["frame"],
        row["id"],
        row["x1"],
        row["y1"],
        row["x2"],
        row["y2"],
        row["type"],
    )
    cam = cam[3:5]
    if _type != "loc":
        continue
    if _id not in bin_ids:
        first_used = "True"
        bin_ids.append(_id)
    else:
        first_used = "False"
    pax_id = "NA"

    if _id in asso_info:
        ffs = list(asso_info[_id])
        for _f in ffs:
            if frame >= _f:
                pax_id = asso_info[_id][_f]
    log_msg = (
        f"LOC: type: DVI camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
        +
        f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} first-used: {first_used} "
        + "partial-complete: NA")
    if cam == '09':
        full_logs_9.append(log_msg)
    elif cam == '11':
        full_logs_11.append(log_msg)
    else:
        full_logs_13.append(log_msg)

# PAX
pax_ids = []
for _, row in tqdm(df_pax.iterrows()):
    cam, frame, _id, x1, y1, x2, y2, _type = (
        row["camera"],
        row["frame"],
        row["id"],
        row["x1"],
        row["y1"],
        row["x2"],
        row["y2"],
        row["type"],
    )
    cam = cam[3:5]
    if _type != "LOC":
        continue
    if _id not in pax_ids:
        first_used = "True"
        pax_ids.append(_id)
    else:
        first_used = "False"
    if "TSO" in _id:
        pax_type = "TSO"
    else:
        pax_type = "PAX"
    log_msg = (
        f"LOC: type: {pax_type} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
        +
        f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: NA first-used: {first_used} "
        + "partial-complete: NA")
    if cam == '09':
        full_logs_9.append(log_msg)
    elif cam == '11':
        full_logs_11.append(log_msg)
    else:
        full_logs_13.append(log_msg)

with open("ata_cam9.txt", "w") as fp:
    fp.write("\n".join(full_logs_9))

with open("ata_cam11.txt", "w") as fp:
    fp.write("\n".join(full_logs_11))

with open("ata_cam13.txt", "w") as fp:
    fp.write("\n".join(full_logs_13))
