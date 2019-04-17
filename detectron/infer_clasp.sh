
export PYTHONPATH="$HOME/Desktop/claspdetectron:$PYTHONPATH"

out_dir="$HOME/dataset/clasp_videos/out/bin"

_dir="$HOME/dataset/clasp_videos/9A"

echo $_dir
for subdir in `ls "${_dir}"`
do
    if [ "$subdir" = "9" ] || [ "$subdir" = "11" ]; then
        echo " " $subdir
        imdir=${_dir}/${subdir}
        outims=${out_dir}/${subdir}
        if ! [ -d "${outims}" ]; then
            echo ${outims} created
            mkdir -p ${outims}
        fi

        outkp=${out_dir}/"kp"/${subdir}

        if ! [ -d ${outkp} ]; then
            echo ${outkp} created
            mkdir -p ${outkp}
        fi

        python tools/infer_simple.py \
            --cfg configs/my_cfg/faster_R-50-FPN_2x.yaml \
            --output-dir ${outims} \
            --kp-thresh 2.0 \
            --image-ext jpg \
            --output-keypoint ${outkp} \
            --wts  '/home/ash/Desktop/clasp/labelling/trained_model_bin/train/clasp_bin/generalized_rcnn/model_final.pkl'\
            ${imdir}
    fi
done


out_dir="$HOME/dataset/clasp_videos/out/pax"

_dir="$HOME/dataset/clasp_videos/9A"

echo $_dir
for subdir in `ls ${_dir}`
do
    if [ "$subdir" = "9" ] || [ "$subdir" = "11" ]; then
        echo " " $subdir
        imdir=${_dir}/${subdir}
        outims=${out_dir}/${subdir}
        if ! [ -d "${outims}" ]; then
            echo ${outims} created
            mkdir -p ${outims}
        fi

        outkp=${out_dir}/"kp"/${subdir}

        if ! [ -d ${outkp} ]; then
            echo ${outkp} created
            mkdir -p ${outkp}
        fi

        python tools/infer_simple.py \
            --cfg configs/my_cfg/pax_faster_R-50-FPN_2x.yaml \
            --output-dir ${outims} \
            --kp-thresh 2.0 \
            --image-ext jpg \
            --output-keypoint ${outkp} \
            --wts  '/home/ash/Desktop/clasp/labelling/trained_model_pax/train/clasp_pax/generalized_rcnn/model_final.pkl'\
            ${imdir}
    fi
done
