export PYTHONPATH='/home/ash/Desktop/clasp/detectron':$PYTHONPATH
python tools/train_net.py \
    --cfg configs/my_cfg/e2e_faster_rcnn_R-50-C4_1x.yaml \
    OUTPUT_DIR /home/ash/Desktop/clasp/weights/detection-weights