from __future__ import division

# from keras.models import Model
# from keras.models import model_from_json
# from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import k_means

# from optimal_k_value_mot import optimal_k_value
from sklearn.cluster import SpectralClustering
from matplotlib.patches import Polygon
from scipy.spatial import distance
# from mpl_toolkits.mplot3d import Axes3D
# from t_SNE_plot import *

# from MeanShift_py.detection_refinement import frame_detection_clustering
# from DHAE import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# import kneed
# from opt_K_elbow import Elbow_opt_K
from tools.data_association import *

# from ae_deconv_28 import normalize
import cv2

# from scipy.misc import imsave
import os
import sys
import glob
import math
from collections import Counter

# from statistics import mode
import random
from numpy.random import seed

seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

# load Model
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def convert_to_30(mask):
    x_t = np.zeros((mask.shape[0], 30, 30), dtype="float")
    x_t[0 : mask.shape[0], 1:29, 1:29] = mask
    # x_t = normalize(x_t)
    return x_t


def prepare_box_data(x_t, im_w, im_h):
    # x_t is the raw data
    x_0 = x_t[:, 2] / im_w
    y_0 = x_t[:, 3] / im_h
    w = x_t[:, 4] / im_w  # np.max(x_t[:, 4])
    h = x_t[:, 5] / im_h  # np.max(x_t[:, 5])
    Cx = (x_t[:, 2] + x_t[:, 4] / 2) / im_w  # np.max((x_t[:, 2] + x_t[:, 4] / .2))
    Cy = (x_t[:, 3] + x_t[:, 5] / 2) / im_h  # np.max((x_t[:, 2] + x_t[:, 4] / .2))
    area = (x_t[:, 4] * x_t[:, 5]) / (im_w * im_h)
    diag = np.sqrt(x_t[:, 4] ** 2 + x_t[:, 5] ** 2) / np.sqrt(im_w ** 2 + im_h ** 2)
    # prepare dim = 8:[Cx,Cy,x,y,w,h,wh,class]
    x_f = np.array([Cx, Cy, w, h])
    # x_f = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])
    x_f = np.transpose(x_f)
    # x_f = normalize(x_f)
    return x_t, x_f


def bandwidth_Nd(feature):
    # function to compute bw for multidimentional latent feature in mean-shift
    bw_vector = []
    for i in range(feature.shape[1]):
        kernel_i = np.var(feature[:, i], axis=0)
        kernel_i = float("{0:.5f}".format(kernel_i))
        if kernel_i == 0:
            # covariance matrix should not be the singular matrix
            kernel_i = kernel_i + 0.00000001
        bw_vector.append(kernel_i)
    bw_vector = np.array(bw_vector)
    return bw_vector


def cluster_mode_det(
    fr,
    latent_feature,
    labels,
    cluster_center,
    det_frame,
    n_angle,
    score_th,
    ID_ind,
    min_cluster_size,
    iou_thr,
):
    # labels comes from k-means (raw id): associate id???
    det_frame = (
        det_frame
    )  # contain all info [CXbox,CYbox, x, y, w, h, classID, angle,fr, score,mask_cx,mask_cy,area,pixels,arc_length]

    final_det, det_frame, ID_ind, cluster_prob_score = cluster_association(
        fr,
        latent_feature,
        det_frame,
        labels,
        cluster_center,
        ID_ind,
        n_angle,
        score_th,
        min_cluster_size,
        iou_thr,
    )
    # print('mask',cluster_i_mask.shape)
    # print('labels',labels)
    # print('j',j)
    # cluster representative selection

    # print(det_frame[:,8])
    return final_det, det_frame, ID_ind, cluster_prob_score


def expand_from_temporal_list(box_all=None, mask_30=None):
    if box_all is not None:
        box_list = [b for b in box_all if len(b) > 0]
        if len(box_list) > 0:
            box_all = np.concatenate(box_list)
        else:
            box_all = None
    if mask_30 is not None:
        mask_list = [m for m in mask_30 if len(m) > 0]
        masks_30 = np.concatenate(mask_list)
    else:
        masks_30 = []
    return box_all, masks_30


def box_mask_overlay(ref_box, ax, im_h, im_w, score_text, color_mask, box_color):
    box_coord = ref_box[2:6].astype(int)
    # mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
    # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
    # mask = np.array(mask >= 0.5, dtype=np.uint8)
    # im_mask = np.zeros((im_h, im_w), dtype=np.float)
    # why only consider bbox boundary for mask???? box can miss part of the object
    x_0 = box_coord[0]
    x_1 = box_coord[0] + box_coord[2]
    y_0 = box_coord[1]
    y_1 = box_coord[1] + box_coord[3]
    # mask transfer on image cooordinate
    # im_mask[y_0:y_1, x_0:x_1] = mask
    # overlay both mask and box on original image
    # im_mask = np.uint8(im_mask * 255)
    # img, contours = cv2.findContours(im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """
    for c in contours:
        polygon = Polygon(
            c.reshape((-1, 2)),
            fill=True, facecolor=color_mask,
            edgecolor='y', linewidth=1.2,
            alpha=0.2)
        ax.add_patch(polygon)
    """
    # show box
    ax.add_patch(
        plt.Rectangle(
            (x_0, y_0),
            box_coord[2],
            box_coord[3],
            fill=False,
            edgecolor=box_color,
            linewidth=2,
            alpha=0.8,
        )
    )
    ax.text(
        box_coord[0],
        box_coord[1] - 2,
        score_text,
        fontsize=12,
        family="serif",
        bbox=dict(facecolor=box_color, alpha=0.5, pad=0, edgecolor="none"),
        color="red",
    )
    return ax

def get_bbox(ref_box, ax, im_h, im_w, score_text, color_mask, box_color):
    box_coord = ref_box[2:6].astype(int)
    x_0 = box_coord[0]
    x_1 = box_coord[0] + box_coord[2]
    y_0 = box_coord[1]
    y_1 = box_coord[1] + box_coord[3]

    # show box
    ax.add_patch(
        plt.Rectangle(
            (x_0, y_0),
            box_coord[2],
            box_coord[3],
            fill=False,
            edgecolor=box_color,
            linewidth=2,
            alpha=0.8,
        )
    )
    ax.text(
        box_coord[0],
        box_coord[1] - 2,
        score_text,
        fontsize=12,
        family="serif",
        bbox=dict(facecolor=box_color, alpha=0.5, pad=0, edgecolor="none"),
        color="red",
    )
    return ax


def cluster_analysis(latent_feature, time_lag, max_instances_at_theta):

    for i in range(max_instances_at_theta, max_instances_at_theta + 3):
        cluster_center, labels, inertia = k_means(latent_feature, n_clusters=i)
        unique, counts = np.unique(labels, return_counts=True)
        if len(np.where(counts > time_lag + 1)) > 0:
            print("cluster size > time_lag+1")
            opt_k = i
        if len(np.where(counts >= time_lag + 1)) == 0:
            opt_k = i
            break
    return opt_k


def temporal_format(boxs):
    # [x,y,w,h,ins_ind,angle,fr,cluster_score]>>>[fr,ins_ind,x,y,w,h,score,class_id,angle]
    formatted_box = np.zeros((boxs.shape[0], 9), dtype="float")
    formatted_box[:, 0] = boxs[:, 6]
    formatted_box[:, 1] = boxs[:, 4]
    formatted_box[:, 2] = boxs[:, 0]
    formatted_box[:, 3] = boxs[:, 1]
    formatted_box[:, 4] = boxs[:, 2]
    formatted_box[:, 5] = boxs[:, 3]
    formatted_box[:, 6] = boxs[:, 7]
    formatted_box[:, 7] = boxs[:, 4]
    formatted_box[:, 8] = 0
    return formatted_box


def tracking_temporal_clustering(
    fr,
    pax_boxs,
    time_lag,
    min_cluster_size,
    iou_thr,
    det_cluster_id,
    ID_ind,
    score_th,
    ax,
    im_h,
    im_w,
):
    # -----------------------------------------------------------------------------
    # **fr - current frame
    # **pax_boxs - temporal window of detections contain both associated and
    #              unassociated sample, format - [fr,ins_ind,x,y,w,h,score,class_id,ID_ind]
    # **min_cluster_size - minimum number of instances in a cluster to consider as
    #                      tracklet
    # **det_cluster_id - already associated detections in loopback frames
    # -----------------------------------------------------------------------------
    temp_window_pbox = []
    k_value = []
    for i in np.linspace(fr - time_lag + 1, fr, num=time_lag):
        # TODO: check that at least one detection at t
        temp_windowb = pax_boxs[np.where(pax_boxs[:, 0] == i), :][0]
        k_value.append(len(temp_windowb[:, 1]))  # max value of instance at t
        if (
            len(det_cluster_id) > 0 and i < fr
        ):  # (fr=6, i=2,3,4,5 has already cluster id initialized detections)
            temp_windowb = det_cluster_id[np.where(det_cluster_id[:, 0] == i)]
        temp_window_pbox.append(temp_windowb)
    temp_window_pbox, _ = expand_from_temporal_list(temp_window_pbox, None)
    # Tracking stops if loop back frames has no detections
    if temp_window_pbox is not None:
        k_value = np.array(k_value)
        # print("number of instances in window:", k_value)
        # prepare mask and box features for DHAE
        temp_pax_boxs, pax_box_norm = prepare_box_data(temp_window_pbox, im_w, im_h)
        latent_feature = pax_box_norm
        max_instances_at_theta = int(np.max(k_value))
        # Analyze latent feature to get optimum k-value
        # print("Maximum PAX Detection from Baseline: ", max_instances_at_theta)
        # print("K value from cluster analysis: ", max_instances_at_theta)
        cluster_center, labels, inertia = k_means(
            latent_feature, n_clusters=max_instances_at_theta
        )
        refined_det, det_cluster_id, ID_ind, cluster_prob_score = cluster_mode_det(
            fr,
            latent_feature,
            labels,
            cluster_center,
            temp_pax_boxs,
            time_lag,
            score_th,
            ID_ind,
            min_cluster_size,
            iou_thr,
        )
        bbox = []
        scores = []
        labels = []
        for i in range(len(refined_det)):
            # save detection to evaluate the overall performance
            cluster_prob_score = refined_det[i, 6]
            if (
                cluster_prob_score >= score_th and sum(refined_det[i][2:5]) > 0
            ):
                refined_det[i, 0] = fr
                # pax_eval.append(refined_det[i])
                # score_text = str(int(refined_det[i, 8])) + "  {:0.2f}".format(
                #     cluster_prob_score
                # )
                box_coord = refined_det[i][2:6].astype(int)
                x_0 = box_coord[0]
                x_1 = box_coord[0] + box_coord[2]
                y_0 = box_coord[1]
                y_1 = box_coord[1] + box_coord[3]
                bbox.append([x_0, y_0, x_1, y_1])
                scores.append(cluster_prob_score)
                labels.append(int(refined_det[i, 8]))

    return det_cluster_id, bbox, scores, labels, ID_ind
