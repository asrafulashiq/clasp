python main.py --file-num exp2 --mode training --cameras cam09 cam11 cam13 \
    --bin-ckpt ~/dataset/ALERT/trained_model/model_cam9_11_13_14.pkl --run-detector \
    --info "" --start-frame 100 --skip-end 0

cp ~/dataset/ALERT/clasp_data/output/run/info_exp2_cam09cam11cam13.csv /data/rpi/logs/
cp ~/dataset/ALERT/clasp_data/output/run/info_exp2_cam09cam11cam13.csv clasp_combine_all/info/


# cd clasp_combine_all/
# python full_integration.py --file-num exp2 --mode training --cameras cam09 cam11 cam13 \
#     --start-frame 100 --skip-end 0


