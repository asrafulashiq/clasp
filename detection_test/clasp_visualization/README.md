### Usage:
---
`python full_integration.py --skip-init [int] --skip-end [int] [--plot] --root [str] --out-dir [str]`


**Argument**

`--skip-init`: how many frames to skip at the beginning (default 2200)

`--skip-end`: how many frames to skip at the beginning (default 100)

`--root`: root path for experiment frames. For example, if we want to work on camera 9 of exp2, the frame path will be `root/exp2/cam09/*.jpg`

`--out-dir`: path to save integrated image file.

`--plot`: whether to save image output to `out-dir`. Omit this flag if you only want ata log file.



The script will create three ata output files for scoring `ata_cam9.txt`, `ata_cam11.txt`, and `ata_cam13.txt`.  
