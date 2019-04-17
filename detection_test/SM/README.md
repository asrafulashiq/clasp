# Tracking Passengers and Divested Items at Airport Checkpoints. (CLASP 2018)

## Detection Algorithm:
We propose a test-time data augmentation and clustering approach to solve the perspective distortion problem in overhead camera scene. We use Mask-RCNN pretrained model trained on COCO datasets to get the test-time augmented (rotation, translation) detection bounding boxes and apply mean-shift clustering technique for final detections.
![detection_visual](https://user-images.githubusercontent.com/42333767/51809556-7156ca80-2267-11e9-8db7-aed620fb3994.png)
![detection_visual1](https://user-images.githubusercontent.com/42333767/51809559-761b7e80-2267-11e9-88d2-cc0b0f53e5c6.png)

## Temporal Clustering Based Data Association and Tracking:
The proposed temporal clustering based online data association and tracking can track multiple instances in the video sequences. Current version of the tracker is unable to handle long term occlusions.
![](tracking10A.gif)
