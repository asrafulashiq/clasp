# python tools/infer_simple.py \
#     --cfg configs/my_cfg/e2e_faster_rcnn_R-50-C4_1x.yaml \
#     --output-dir /home/ash/Desktop/clasp_detection/output/detectron-output \
#     --image-ext jpg \
#     --wts /home/ash/Desktop/clasp_detection/weights/detection-weights/train/clasp_detect/generalized_rcnn/model_final.pkl \
#     /home/ash/Desktop/images_track_clasp/tmp

export PYTHONPATH="$HOME/clasp/detectron:$PYTHONPATH"

wts_bin="$HOME/clasp/labelling/trained_model_bin/train/clasp_bin/generalized_rcnn/model_final.pkl"

im_dir="/media/drive/alert_frames/exp3/cam09"
outims="/media/drive/tmp_exp3"


python tools/infer_simple.py \
            --cfg configs/my_cfg/faster_R-50-FPN_2x.yaml \
            --output-dir ${outims} \
            --image-ext jpg \
            --wts ${wts_bin} \
            ${im_dir}
