import os
import sys
import pandas as pd
import parse


def read_ata_log_to_df(filename: str) -> pd.DataFrame:
    """read ata log file and convert it to pandas dataframe

    One sample line:
        LOC: type: DVI camera-num: 13 frame: 1400 time-offset: 140.00 \
        BB: 1739, 184, 1917, 452 ID: P1_A PAX-ID: P1 partial-complete: Partial \
        left-behind: False empty: False \
        description: BROWN CYLINDER ROD, STICK AND A BLACK BAG ON IT
    """
    line_format = (
        "LOC: type: {type:w} camera-num: {camera_num:w} frame: {frame:d} time-offset: {time_offset:f} "
        "BB: {x1:d}, {y1:d}, {x2:d}, {y2:d} ID: {bin_id:w} PAX-ID: {pax_id:w} partial-complete: {}"
        "left-behind: {left_behind:w} empty: {is_empty:w} "
        "description:{}")
    line_parse = parse.compile(line_format)

    memory = []
    columns = [
        'frame', 'camera_num', 'time_offset', 'x1', 'y1', 'x2', 'y2',
        'is_empty', 'type'
    ]
    with open(filename, 'r') as f:
        for line in f:
            res = line_parse.parse(line)
            if res and res['type'] == 'DVI':
                memory.append(list(res[x] for x in columns))
    if not memory:
        raise ValueError("No gt here")
    df = pd.DataFrame(memory, columns=columns)
    df['is_empty'] = df['is_empty'].astype(bool)
    return df


if __name__ == '__main__':
    filename = os.path.expanduser(
        "~/dataset/ALERT/annotation_gt/exp2_train/10fps_EmptyBins/cam11exp2-training-logfile.txt"
    )
    df = read_ata_log_to_df(filename)
    print(df.head())
