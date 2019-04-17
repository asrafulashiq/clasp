
export PYTHONPATH="$HOME/clasp/detectron:$PYTHONPATH"

out_dir="$HOME/dataset/clasp_data/out/bin"

_dir="$HOME/dataset/clasp_data/10A"

wts_bin="$HOME/clasp/labelling/trained_model_bin/train/clasp_bin/generalized_rcnn/model_final.pkl"

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
            --image-ext png \
            --output-keypoint ${outkp} \
            --wts ${wts_bin} \
            ${imdir}
    fi
done


# out_dir="$HOME/dataset/clasp_videos/out/pax"

# _dir="$HOME/dataset/clasp_videos/10A"

# wts_pax="${HOME}/clasp/labelling/trained_model_pax/train/clasp_pax/generalized_rcnn/model_final.pkl"

# echo $_dir

# for subdir in `ls ${_dir}`
# do
#     if [ "$subdir" = "9" ] || [ "$subdir" = "11" ]; then
#         echo " " $subdir
#         imdir=${_dir}/${subdir}
#         outims=${out_dir}/${subdir}
#         if ! [ -d "${outims}" ]; then
#             echo ${outims} created
#             mkdir -p ${outims}
#         fi

#         outkp=${out_dir}/"kp"/${subdir}

#         if ! [ -d ${outkp} ]; then
#             echo ${outkp} created
#             mkdir -p ${outkp}
#         fi

#         python tools/infer_simple.py \
#             --cfg configs/my_cfg/pax_faster_R-50-FPN_2x.yaml \
#             --output-dir ${outims} \
#             --kp-thresh 2.0 \
#             --image-ext png \
#             --output-keypoint ${outkp} \
#             --wts ${wts_pax} \
#              ${imdir}
#     fi
# done
