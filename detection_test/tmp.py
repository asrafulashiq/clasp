import os
from pathlib import Path

root = '/run/media/ash/7ee91e7f-d8ac-4f0a-8e6a-750d09659980/ALERT' + \
    '/clasp_data/output/run/exp2_train/cam13/'
root = Path(root)

files = sorted(root.iterdir())

for fp in files:
    name = fp.stem
    fnum = int(name)
    if fnum % 2 == 0:
        os.remove(str(fp))

