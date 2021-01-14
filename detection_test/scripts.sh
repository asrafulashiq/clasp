python -m pdb main.py --start_frame 2500 --cameras cam09 cam11 \
    --bin_ckpt ~/dataset/ALERT/trained_model/weight_640x360.pkl \
    --size 640x360 --out_suffix _640x360 --spatial_scale_mul 1 \
    --temporal_scale_mul 3 --run_detector --out_suffix _param-2
