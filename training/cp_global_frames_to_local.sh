# copy frames from global to my local

out_path="${HOME}/dataset/ALERT/alert_frames/20191024/fps-10/exp2_train"
in_path="${HOME}/dataset/ALERT/gt_annotations/10fps_frames/exp2_train"

for each_cam in $(ls ${in_path}/); do
    mkdir -p ${out_path}/${each_cam}

    for fp in $(ls ${in_path}/${each_cam}/); do
        ww=$(cut -d'.' -f1 <<<"$fp")
        qq=$(cut -d'_' -f2 <<<"$ww")

        echo $qq
        out_name=$(printf "%05d" "$qq").jpg

        cp ${in_path}/${each_cam}/${fp} ${out_path}/${each_cam}/${out_name}

    done

done
