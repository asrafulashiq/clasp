
python train.py -a ../../labelling/annotations/fullsize_exp1_exp2_train_exp3cam09_cam11_cam13_cam14_cam20.json --size 420x240 -o weight_420x240_aug.pkl  --epoch 10 --lr 1e-4 --batch_size 8

python train.py -a ../../labelling/annotations/fullsize_exp1_exp2_train_exp3cam09_cam11_cam13_cam14_cam20.json --size 640x360 -o weight_640x360_aug.pkl --ckpt ~/dataset/ALERT/trained_model/weight_420x240_aug.pkl  --epoch 20 --lr 1e-4 --batch_size 8

python train.py -a ../../labelling/annotations/fullsize_exp1_exp2_train_exp3cam09_cam11_cam13_cam14_cam20.json --size 1280x720 -o weight_1280x720_aug.pkl --ckpt ~/dataset/ALERT/trained_model/weight_640x360_aug.pkl  --epoch 20 --lr 5e-3 --batch_size 4

for cam in cam09 cam11; do
  for size in "640x360" "1280x720"; do
  python test.py --ckpt ~/dataset/ALERT/trained_model/weight_${size}_aug.pkl      --size ${size} --write_folder ~/datasets/ALERT/out_detection_test/exp2_${cam}_${size}_aug --folder ~/dataset/ALERT/alert_frames/20191024/exp2_train/${cam}
  done
done




