import numpy as np
import pandas as pd

# file_path = "/run/media/ash/7ee91e7f-d8ac-4f0a-8e6a-750d09659980/ALERT/clasp_data/output/run/info.csv"
file_path = "/home/ash/Desktop/info.csv"

df = pd.read_csv(
    file_path,
    sep=",",
    header=None,
    names=["file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2", "type", "msg"],
    index_col=None,
)

print(df.head())

df_new = df.copy()
df_new['frame'] = df['frame'] + 1

df_comb = pd.concat((df, df_new))
df_comb = df_comb.sort_values('frame')
df_comb = df_comb[df_comb['type'] == 'loc']

# df_comb = df

# up-scale video resolution : 3x
df_comb["x1"] = 3 * df_comb["x1"]
df_comb["y1"] = 3 * df_comb["y1"]
df_comb["x2"] = 3 * df_comb["x2"]
df_comb["y2"] = 3 * df_comb["y2"]

df_comb.to_csv("/home/ash/Desktop/info_all.csv", header=None, index=False)