export PYTHONPATH="$HOME/clasp/detectron:$PYTHONPATH"

python tools/train_net.py \
    --cfg configs/my_cfg/faster_R-50-FPN_2x.yaml \
    OUTPUT_DIR $HOME/dataset/clasp_annotations/trained_model_bin