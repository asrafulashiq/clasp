import numpy as np
import pandas as pd

file_path = "/run/media/ash/7ee91e7f-d8ac-4f0a-8e6a-750d09659980/ALERT/clasp_data/output/run/info.csv"

df = pd.read_csv(
    file_path,
    sep=",",
    header=None,
    names=["file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2", "type", "msg"],
    index_col=None,
)

df['x1'] = df['x1'] * 3
df['y1'] = df['y1'] * 3
df['x2'] = df['x2'] * 3
df['y2'] = df['y2'] * 3

new_df = df.copy()
new_df['frame'] = new_df['frame'] + 1

df_all = pd.concat((df, new_df))

print(df_all.head())

df_all = df_all.sort_values(by="frame")
df_all = df_all[df_all["type"]=="loc"]

df_all.to_csv("/home/ash/Desktop/info_bin.csv", index=None)