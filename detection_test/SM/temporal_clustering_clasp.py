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
from mpl_toolkits.mplot3d import Axes3D
from t_SNE_plot import *

# from MeanShift_py.detection_refinement import frame_detection_clustering
# from DHAE import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# import kneed
# from opt_K_elbow import Elbow_opt_K
from data_association import *

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


def plot_latent_feature(x_test, encoded_imgs, decoded_imgs, img_y, img_x, names):
    num_images = 20
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    minval, maxval = encoded_imgs.min(), encoded_imgs.max()

    plt.figure(figsize=(30, 4))

    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(
            encoded_imgs[image_idx].reshape(4, 8), cmap="hot", vmin=minval, vmax=maxval
        )
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(
        "/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-09/latent_feature/"
        + names[-10:],
        dpi=300,
    )
    plt.close()


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


def visualize_box_reconstruction(x_test, encoded_imgs, decoded_imgs, names):
    # visualize compressed encoded feature
    num_images = 30
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    plt.figure(figsize=(30, 4))
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.stem(x_test[image_idx])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.stem(encoded_imgs[image_idx])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.stem(decoded_imgs[image_idx])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(
        "/media/siddique/Data/CLASP2018/cluster_result/6A/cam9/final_model/box_recon/"
        + names[-10:],
        dpi=300,
    )
    plt.close()


def plot_mask_cluster(
    x_test, labels, max_instances_at_theta, img_y, img_x, names, path
):
    ids, counts = np.unique(labels, return_counts=True)
    if max_instances_at_theta > 4:
        plt.figure(figsize=(40, max_instances_at_theta))

    if max_instances_at_theta <= 4:
        plt.figure(figsize=(30, max_instances_at_theta))

    for i in range(ids.shape[0]):
        cluster_i = x_test[np.where(labels == i), :, :][0]
        if i == 0:
            for j in range(counts[i]):
                ax = plt.subplot(max_instances_at_theta, counts.max(), j + 1)
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 1:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 2:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 2 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 3:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 3 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 4:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 4 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 5:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 5 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 6:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 6 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 7:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 7 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if i == 8:
            for j in range(counts[i]):
                ax = plt.subplot(
                    max_instances_at_theta, counts.max(), 8 * counts.max() + j + 1
                )
                plt.imshow(cluster_i[j].reshape(img_y, img_x), cmap="hot")
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(path + "mask_cluster/" + names[-10:], dpi=300)
    plt.close()


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


def concat_images(img1, img2, names, path):
    img_final = np.hstack([img1, img2])
    im = Image.fromarray(img_final)
    im.save(path + "/all_box_on_img/" + names[-10:])


def denormalized_box(x_t, im_w, im_h):
    w = x_t[:, 2] * im_w  # / 1920#np.max(x_t[:, 4])
    h = x_t[:, 3] * im_h  # / 1080#np.max(x_t[:, 5])
    x_0 = x_t[:, 0] * im_w - w / 2
    y_0 = x_t[:, 1] * im_h - h / 2
    x_f = np.array([x_0, y_0, w, h])  # , x_0, y_0, area,diag])
    x_f = np.transpose(x_f)
    return x_f


def all_det_on_image(
    img1, input_box_f, decoded_box_f, color, names, unique, path
):  # decode_box_f: (n_sample,4), input_box_f: (n_sample,9)
    # decoded_box_f = denormalized_box(decoded_box_f)
    for i in range(input_box_f.shape[0]):
        input_box = input_box_f[i, :]

        img1 = cv2.rectangle(
            img1,
            (int(input_box[2]), int(input_box[3])),
            (int(input_box[2] + input_box[4]), int(input_box[3] + input_box[5])),
            color,
            4,
        )
        # img2 = cv2.rectangle(img2, (int(decoded_box[0]),int(decoded_box[1])),\
        # (int(decoded_box[0]+decoded_box[2]), int(decoded_box[1]+decoded_box[3])),color, 5 )
    for i in range(len(unique)):
        decoded_box = decoded_box_f[i, :]
        cluster_score = decoded_box[6]
        score_text = " {:0.2f}".format(cluster_score)
        img1 = cv2.rectangle(
            img1,
            (int(decoded_box[2]), int(decoded_box[3])),
            (
                int(decoded_box[2] + decoded_box[4]),
                int(decoded_box[3] + decoded_box[5]),
            ),
            (0, 0, 255),
            4,
        )
        cv2.putText(
            img1,
            score_text,
            (int(decoded_box[2]), int(decoded_box[3])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    im = Image.fromarray(img1)
    im.save(path + "/all_box_on_img/" + names[-10:])
    # concat_images(img1, img2, names,path)


def plot_tSNE(X, labels, path):
    Y = tsne(X, 2, 104, 20.4)
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], 30, labels, cmap="hot")
    plt.savefig(path, dpi=300)
    plt.close()


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


#
# Select the temporal window for clustering : use t-5
#


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
    temp_window_pmask = []
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
        # temp_windowm = pax_mask[np.where(pax_boxs[:, 0] == i), :, :][0]
        # temp_window_pmask.append(temp_windowm)
    temp_window_pbox, _ = expand_from_temporal_list(temp_window_pbox, None)
    # Tracking stops if loop back frames has no detections
    if temp_window_pbox is not None:
        k_value = np.array(k_value)
        print("number of instances in window:", k_value)
        #
        # prepare mask and box features for DHAE
        #
        temp_pax_boxs, pax_box_norm = prepare_box_data(temp_window_pbox, im_w, im_h)
        # x_test = temp_window_pmask
        # mask_x_test = np.reshape(x_test, [x_test.shape[0], img_y, img_x, 1])
        # fr_pax_box = pax_boxs[np.where(pax_boxs[:, 0] == fr), :][0]
        # fr_mask = pax_mask[np.where(pax_boxs[:, 0] == fr), :, :][0]
        #
        # bottleneck feature and reconstruction for temporal window t-10
        #
        # decoded_imgs, decoded_box = final_model.predict([mask_x_test, pax_box_norm])
        # latent_feature = bottleneck_model.predict([mask_x_test, pax_box_norm])
        latent_feature = pax_box_norm
        # plot_latent_feature(fr_x_test, latent_feature,decoded_imgs,img_y,img_x,names)
        #
        # Compute optimum k-values in k-means for target clusters at current frame
        #
        # max_instances_at_theta = int(temp_pax_boxs[:,1].max())
        max_instances_at_theta = int(np.max(k_value))
        #
        # Analyze latent feature to get optimum k-value
        #
        print("Maximum PAX Detection from Baseline: ", max_instances_at_theta)
        # from sklearn.cluster import SpectralClustering
        # model=SpectralClustering(n_clusters=max_instances_at_theta,affinity='nearest_neighbors',n_neighbors=10,assign_labels='kmeans')
        # labels = model.fit_predict(latent_feature)
        # Number of clusters  = maximum nuber of objects (FP + TP) at any of the 20 different angles
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2,init='random',random_state=0)
        # dat_proj = tsne.fit_transform(latent_feature)
        # latent_feature = dat_proj
        # max_instances_at_theta = cluster_analysis(latent_feature, time_lag, max_instances_at_theta)
        # max_instances_at_theta = optimal_k_value(latent_feature, max_instances_at_theta,temp_pax_boxs,time_lag)
        # print('K value from Sillhoutee: ', max_instances_silhoutee)
        # max_instances_elbow = Elbow_opt_K(latent_feature, max_instances_at_theta)
        print("K value from cluster analysis: ", max_instances_at_theta)
        cluster_center, labels, inertia = k_means(
            latent_feature, n_clusters=max_instances_at_theta
        )  # ,\
        # ,sample_weight=temp_pax_boxs[:,6],n_init=max_instances_at_theta,max_iter=300,precompute_distances='auto',algorithm='auto',tol=1e-4,random_state=10) #init='random',
        # Getting the cluster labels, cluster_center
        # labels_id = temp_pax_boxs[:,8]
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
        # det_cluster_id,_ = expand_from_temporal_list(det_cluster_id)
        # all_det_on_image(img1,det_at_t_pbox,refined_det,box_color,names,unique, \
        #'/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-10')
        # Select best mask from the clusters
        # plot clustered result using t-SNE
        # plot_tSNE(latent_feature,labels)
        # plot_mask_cluster(fr_x_test, labels,max_instances_at_theta, img_x, img_y, names)
        # show clustered result in lower dimension
        # tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
        """
        Y = tsne(X, 2, 18, 20.0)
        plt.scatter(Y[:, 0], Y[:, 1], 30, labels,cmap='hot')
        """
        # unique, counts = np.unique(det_cluster_id[:,8], return_counts=True)'
        """
        if (len(unique) < 1):
            plot_mask_cluster(fr_x_test, labels, len(unique), img_x, img_y, names,\
                               '/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-11/')
        """
        for i in range(len(refined_det)):
            # save detection to evaluate the overall performance
            cluster_prob_score = refined_det[i, 6]
            if (
                cluster_prob_score >= score_th and sum(refined_det[i][2:5]) > 0
            ):  # and refined_det[i, 8]>0
                refined_det[i, 0] = fr
                pax_eval.append(refined_det[i])
                color_mask = color[
                    int(refined_det[i, 8])
                ]  # np.array([0, 1, 0], dtype='float32')  # green
                box_color = color[int(refined_det[i, 8])]
                score_text = str(int(refined_det[i, 8])) + "  {:0.2f}".format(
                    cluster_prob_score
                )
                ax = box_mask_overlay(
                    refined_det[i], ax, im_h, im_w, score_text, color_mask, box_color
                )
    return fig, pax_eval, ax, det_cluster_id, ID_ind


## Call tracking_temporal_clustering() function to get trajectory on image ####
# initialize color arrays for visualization
number_of_colors = 800  # should be greater than possible IDs
color = [
    "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    for i in range(number_of_colors)
]
# parameters for tracker
time_lag = 5
score_th = 0.15
min_cluster_size = 3
iou_thr = 0.1
# video sequence detections
boxs = np.loadtxt(
    "/home/ashraful/Downloads/det10ACam11PersonMASK_30FPS_cluster_score",
    delimiter=",",
)
# masks = masks[np.where(boxs[:, 8] == 0 ), :,:][0]
# boxs = boxs[np.where(boxs[:, 8] == 0 ), :][0]
boxs = temporal_format(boxs)
# masks = masks[np.where(boxs[:, 6] >= 0.5 ), :,:][0]
boxs = boxs[np.where(boxs[:, 6] >= 0.4), :][0]

pax_boxs = boxs[np.where(boxs[:, 7] == 1), :][0]
# pax_mask = masks[np.where(boxs[:, 7] == 1), :, :][0]
# use trained CAE to get embedded feature for clustering
# final_model, self_express_model, bottleneck_model, mask_encoder, box_encoder = load_model()
# final_model,bottleneck_model,box_encoder_model = Create_DHAE_Model(img_y,img_x)
# checkpoint_path = '/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weight_motv3/cp-1000.ckpt'#'/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weightv2/cp-1000.ckpt'
# final_model.load_weights(checkpoint_path)

## frame by frame qualitative tracking evaluation
# path = "/media/siddique/RemoteServer/CLASP/CLASP_Data/Data_GT/ourdataset/exp10A/cam11/30FPS/*.png"
path = "/home/ashraful/dataset/clasp_data/10A/11/*.png"

files = glob.glob(path)
files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
fr = 1
pax_eval = []
bag_eval = []
det_cluster_id = []
ID_ind = 1
pax_boxs_window = []
for names in files:
    # detection for a frame
    fr_pax_box = pax_boxs[np.where(pax_boxs[:, 0] == fr), :][0]
    if len(fr_pax_box) > 0:
        # initial frame with detection
        print(names)
        # To overlay the boxs and masks on original image
        im = cv2.imread(names)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_h, im_w, _ = im.shape
        # define figure
        dpi = 200
        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(im)
        # add new frame with previously tracked window
        if len(det_cluster_id) > 0:
            pax_boxs_prev = np.vstack([det_cluster_id, fr_pax_box])
            # pax_boxs_prev, _ = expand_from_temporal_list(pax_boxs_window, None)
        else:
            pax_boxs_window.append(fr_pax_box)
            pax_boxs_prev, _ = expand_from_temporal_list(pax_boxs_window, None)
        # wait for loop back frames
        if fr >= pax_boxs_prev[:, 0][0] + time_lag - 1:
            fig, pax_track, ax, det_cluster_id, ID_ind = tracking_temporal_clustering(
                fr,
                pax_boxs_prev,
                time_lag,
                min_cluster_size,
                iou_thr,
                det_cluster_id,
                ID_ind,
                score_th,
                ax,
                im_h,
                im_w,
            )
            print("Total tracked ID:", ID_ind)
            pax_eval.append(pax_track)
            fig.savefig(
                "/home/ashraful/Desktop/tmp/exp10A/" + names[-10:],
                dpi=dpi,
            )
        plt.close("all")
    fr += 1

pax_track = np.array(pax_eval)
np.savetxt(
    "~/Desktop/tmp/exp10A_track",
    pax_eval,
    fmt="%.4e",
    delimiter=",",
)

