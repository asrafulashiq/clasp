from __future__ import division
import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means

def get_iou(a,b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # format conversion [x,y,w,h]>>[x1,y1,x2,y2]

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def one2all_iou(one,all):
    iou_vector = np.array([get_iou(i, j, epsilon=1e-5) for i in one for j in all]).reshape(one.shape[0], all.shape[0])
    return iou_vector

def pairwise_iou_non_diag(a):
    iou_matrix = np.array([get_iou(i, j, epsilon=1e-5) for i in a for j in a]).reshape(a.shape[0], a.shape[0])
    xu, yu = np.triu_indices_from(iou_matrix, k=1)
    xl, yl = np.tril_indices_from(iou_matrix, k=-1)
    x = np.concatenate((xl, xu))
    y = np.concatenate((yl, yu))
    non_diag_ious = iou_matrix[(x, y)].reshape(iou_matrix.shape[0], iou_matrix.shape[0]-1)

    return non_diag_ious


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou

def better_np_unique(arr):
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, inverse, counts = np.unique(arr,
        return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, inverse, counts

def pair_dist_ind(X,Y):
    dist = pairwise_distances(X, Y, metric='euclidean') #X by Y size
    centroid_ind = np.sum(dist,axis=1).argmin()
    return centroid_ind

def final_det_mask(cluster_Q, cluster_i_mask, cluster_i_embed, cluster_center, time_lag):
    #
    # final tracklet member selection from the survived cluster
    #
    cluster_prob_score = np.sum(cluster_Q[:, 6]) / time_lag
    score = np.around(cluster_Q[:, 6], decimals=2)  # cluster score rounded upto two decimal points
    print('cluster score', cluster_prob_score)
    refined_det = cluster_Q[np.where(score == score.max())]  # single or multiple detections might be the representtive members of a cluster
    # refined_det = cluster_Q[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    # feature_center_ini = cluster_i_shifted_points[np.where(score == score.max())]
    refined_mask = cluster_i_mask[np.where(score == score.max())]
    refined_embed = cluster_i_embed[np.where(score == score.max())]
    # refined_mask = cluster_i_mask[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    # refined_embed = cluster_i_embed[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    if (len(refined_det) > 1):
        centroid_ind = pair_dist_ind(refined_embed, cluster_center)
        refined_det = refined_det[centroid_ind]  # select closest one from multiple representative
        refined_mask = refined_mask[centroid_ind, :, :]
        refined_det[6] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
    else:
        refined_det = refined_det.flatten()
        refined_det[6] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability

    return refined_det, refined_mask


def cluster_association(fr,latent_feature, det_frame, labels,
                        cluster_center, ID_ind, time_lag,score_th,min_cluster_size,iou_thr):
    final_det = []
    #final_mask = []
    labels_unique, counts = np.unique(labels, return_counts=True)
    for j in labels_unique: #cluster wise loop
        # filter noise by selecting single detection foe a cluster at an angle
        # ignore repeated angle sample in det_frame
        track_new = 0
        track_associated = 0
        cluster_Q = det_frame[np.where(labels == j), :][0] # all angles of Fa cluster
        cluster_i_embed = latent_feature[np.where(labels == j), :][0]
        #cluster_i_mask = det_at_t_pmask[np.where(labels == j), :, :][0]
        cluster_label = cluster_Q[:,8]
        cluster_prob_score = np.sum(cluster_Q[:, 6]) / time_lag
        cluster_size = cluster_Q[:, 8].shape[0]

        # Case1: All samples are unassociated [cluster size <= temporal window]
        boxs = np.transpose(np.array(
            [cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4], cluster_Q[:, 3] + cluster_Q[:, 5]]))
        iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
        avg_iou = np.mean(iou_matrix_non_diag, axis=1)
        if (sum(cluster_Q[:, 8]) == 0 and cluster_size <= time_lag and
                cluster_size >= min_cluster_size and len(cluster_Q[:, 8][np.where(avg_iou>=iou_thr)])>0):
           # All cluster ssamples are unassociated (zero id)
           # ID should not be updated for outlier
           #dist = pairwise_distances(cluster_i_embed, metric='euclidean') TODO: pairwise cosine similarity score
           #avg_dist = np.mean(dist, axis=1)
           #
           # Cluster sample analysis: good - initialized new id or bad - TODO: robust method to separate good and bad sample
           #
        #    print('average_IoU of cluster samples:',avg_iou)
        #    print('Cluster Label Before Association', cluster_Q[:, 8])
           cluster_Q[:, 8][np.where(avg_iou>=iou_thr)] = ID_ind  # initialize all detections in a cluster with cluster ID
           det_frame[np.where(labels == j), :] = cluster_Q
        #    print('new trajectory', ID_ind)
        #    print('new trajectory labels after association', det_frame[np.where(labels == j), 8])
        #    print('Frame Pattern',cluster_Q[:, 0])
           ID_ind += 1
           track_new = 1

           #refined_det, refined_mask = final_det_mask(cluster_Q, cluster_i_mask,
            #                                          cluster_i_embed, cluster_center,
             #                                         time_lag)
           # TODO: obtain final mask and det from the right edge of the temporal window
           refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
           #refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
           #final_mask.append(refined_mask.reshape(28, 28))
           final_det.append(refined_det)

        # Case2: All samples are unassociated [cluster size > temporal window]
        elif (sum(cluster_Q[:, 8]) == 0 and cluster_size > time_lag):
        #    print('Find Merged Cluster...')
           #boxs = np.transpose(np.array([cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4], cluster_Q[:, 3] + cluster_Q[:, 5]]))
           #iou_matrix = pairwise_iou(boxs)
           #avg_iou = np.mean(iou_matrix, axis=1)
           #print('average_IoU of cluster samples:',avg_iou)
           # Splitting clusters using frame pattern in temporal window
           frame_unique, count_cwc = np.unique(cluster_Q[:,0], return_counts=True)
           center_cwc, label_cwc, inertia_cwc = k_means(cluster_i_embed, n_clusters=count_cwc.max())
           for k in np.unique(label_cwc):
               cluster_cwc = cluster_Q[np.where(label_cwc==k)]
               cluster_cwc_size = len(cluster_cwc[:,0])
               boxs = np.transpose(np.array(
                   [cluster_cwc[:, 2], cluster_cwc[:, 3], cluster_cwc[:, 2] + cluster_cwc[:, 4],
                    cluster_cwc[:, 3] + cluster_cwc[:, 5]]))
               iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
               avg_iou = np.mean(iou_matrix_non_diag, axis=1)
               if(cluster_cwc_size>=min_cluster_size and len(cluster_cwc[:, 8][np.where(avg_iou>=iou_thr)])>0):#sum(cluster_cwc[:,6])/time_lag >=score_th
                #   print('Cluster Label Before Association',cluster_cwc[:,8])
                  cluster_cwc[:,8][np.where(avg_iou>=0.3)] = ID_ind  # initialize all detections in a cluster with cluster ID
                  cluster_Q[np.where(label_cwc==k)] = cluster_cwc
                  #splitted_mask = cluster_i_mask[np.where(cluster_Q[:,8] == ID_ind)]
                  splitted_embed = cluster_i_embed[np.where(cluster_Q[:,8] == ID_ind)]
                  #refined_det, refined_mask = final_det_mask(cluster_cwc, splitted_mask, splitted_embed,
                   #                                          center_cwc,
                    #                                         time_lag)
                  # TODO: obtain final mask and det from the last corner of the temporal window
                  refined_det = cluster_cwc[np.where(cluster_cwc[:, 0] == cluster_cwc[:, 0].max())][0]
                  #refined_mask = splitted_mask[np.where(cluster_cwc[:, 0] == cluster_cwc[:, 0].max())][0]
                  #final_mask.append(refined_mask.reshape(28, 28))
                  final_det.append(refined_det)
                #   print('new trajectory', ID_ind)
                #   print('new trajectory labels after association', cluster_cwc[:,8])
                  ID_ind += 1
                  track_new = 1

               det_frame[np.where(labels == j), :] = cluster_Q

        #
        # Cluster contain both associated and unassociated samples > [6,6,6,6,0] or [6 6 0 0 0] or [6 1 6 1 6 1 6 1 0 0]
        # what about [6,6,6,6,6]; new cluster might have zero id sample or not
        # Use cluster_new = [0,1] (in all zero case) identifier to consider new cluster with no zero id
        elif (len(cluster_Q[np.where(cluster_Q[:, 8] == 0)]) > 0 and
                sum(cluster_Q[:, 8])>0 and cluster_size>=min_cluster_size):#cluster_prob_score >= score_th
            # from statistics import mode, mode(cluster_Q[:,8])
            #cluster_label = cluster_Q[:, 8]
            # print('Cluster (asso + unasso)',cluster_label)
            # Case3: cluster size > time lag:  [6 1 6 1 6 1 6 1 0 0]
            if(len(cluster_Q)>time_lag):
                # Find best match for each newly added samples in the cluster
                #for i in range(len(cluster_Q[np.where(cluster_Q[:, 8] == 0)])):
                    #dist = pairwise_distances(cluster_i_embed[np.where(cluster_Q[:, 8] == 0)][i].reshape(1,32), cluster_i_embed)[0]
                #boxs = np.transpose(np.array([cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4], cluster_Q[:, 3] + cluster_Q[:, 5]]))
                #iou_matrix = pairwise_iou(boxs)
                #avg_iou = np.mean(iou_matrix, axis=1)
                #print('Average Pairwise IoU:', avg_iou)
                # Apply kmeans again on merged cluster using frame pattern in temporal window
                frame_unique, count_cwc = np.unique(cluster_Q[:, 0], return_counts=True)
                center_cwc, label_cwc, inertia_cwc = k_means(cluster_i_embed, n_clusters=count_cwc.max())
                for k in np.unique(label_cwc):
                    cluster_cwc = cluster_Q[np.where(label_cwc == k)]
                    cluster_cwc_size = len(cluster_cwc[:, 0])
                    cluster_cwc_label = cluster_cwc[:,8]
                    boxs = np.transpose(np.array(
                        [cluster_cwc[:, 2], cluster_cwc[:, 3], cluster_cwc[:, 2] + cluster_cwc[:, 4],
                         cluster_cwc[:, 3] + cluster_cwc[:, 5]]))
                    iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
                    avg_iou = np.mean(iou_matrix_non_diag, axis=1)
                    if (cluster_cwc_size >= min_cluster_size and len(cluster_cwc[:, 8][np.where(avg_iou>=iou_thr)])):  # sum(cluster_cwc[:,6])/time_lag >=score_th
                    #if (sum(cluster_cwc[:, 6]) / time_lag >= score_th): # Consider splitted cluster with score >=0.5
                        uni, cnt = np.unique(cluster_cwc_label, return_counts=True)
                        # print('trajectory labels before association', cluster_cwc[:, 8])
                        if ( sum(cluster_cwc_label)==0 and sum(cluster_cwc[:, 6])/time_lag >=score_th):  # New object in FOV
                            cluster_cwc[:, 8][np.where(avg_iou >= iou_thr)] = ID_ind
                            # print('new trajectory', ID_ind)
                            # print('new trajectory labels after association', cluster_cwc[:, 8])
                            ID_ind += 1
                        elif(sum(cluster_cwc_label) > 0):
                            non_zero_id = cluster_cwc_label[cluster_cwc_label.nonzero()]
                            uni, cnt = np.unique(non_zero_id, return_counts=True)
                            cluster_cwc[:, 8][np.where(avg_iou >= iou_thr)] = uni[np.where(cnt == cnt.max())].max()

                        # print('trajectory labels after association', cluster_cwc[:, 8])
                        cluster_Q[np.where(label_cwc == k)] = cluster_cwc
                        #splitted_mask = cluster_i_mask[np.where(label_cwc == k)]
                        #splitted_embed = cluster_i_embed[np.where(label_cwc == k)]
                        #refined_det, refined_mask = final_det_mask(cluster_cwc, splitted_mask, splitted_embed,
                         #                                          center_cwc,
                          #                                         time_lag)
                        # TODO: obtain final mask and det from the last corner of the temporal window
                        refined_det = cluster_cwc[np.where(cluster_cwc[:,0]==cluster_cwc[:,0].max())][0]
                        #refined_mask = splitted_mask[np.where(cluster_cwc[:,0]==cluster_cwc[:,0].max())][0]
                        #final_mask.append(refined_mask.reshape(28, 28))
                        final_det.append(refined_det)
                        # print('After Splitted Cluster Association:', cluster_cwc[:, 8])

                    det_frame[np.where(labels == j), :] = cluster_Q
                    track_associated = 1

                #dist = pairwise_distances(cluster_i_embed, metric='euclidean')
                #avg_dist = np.mean(dist, axis=1)
                #print('Average Pairwise Distance:', avg_dist)

                #uni, cnt = np.unique(cluster_label, return_counts=True)
                #cluster_label[np.where(avg_iou>=0.2)] = uni.max()#cluster_label[np.where(avg_iou==avg_iou.max())][0]
            else:
                #
                # Case4: Cluster size <= Time Lag
                #
                # id_value,count_id = np.unique(cluster_label,return_counts=True)
                # id_cluster = id_value[np.argmax(count_id)]# TODO: all member in cluster already have different iD???
                #uniq, freq = np.unique(cluster_label, return_counts=True)
                #cluster_label[np.where(cluster_label == 0)] = max(cluster_label, key=Counter(cluster_label).get)  # [np.where(cluster_label == 0)]
                boxs = np.transpose(np.array(
                    [cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4],
                     cluster_Q[:, 3] + cluster_Q[:, 5]]))
                iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
                avg_iou = np.mean(iou_matrix_non_diag, axis=1)
                uni, cnt = np.unique(cluster_label, return_counts=True)

                if(len(cluster_label[np.where(cluster_label==0)])>=min_cluster_size and len(cluster_Q[:, 8][np.where(avg_iou>=0.3)])>0
                        and uni[np.where(cnt == cnt.max())].max()==0 and sum(cluster_Q[:, 6])/time_lag >=score_th):
                    # Initiate new object trajectroy when all IDs are zero with cluster score>=0.5
                    cluster_label[np.where(avg_iou >= iou_thr)] = uni[np.where(cnt == cnt.max())].max()  # cluster_label[np.where(avg_iou == avg_iou.max())][0] uni[np.argmax(cnt)]
                    # print('new trajectory labels before association', cluster_label)
                    cluster_label[::]=ID_ind
                    # print('new trajectory', ID_ind)
                    # print('new trajectory labels after association', cluster_label)
                    ID_ind += 1
                else:
                    non_zero_id = cluster_label[cluster_label.nonzero()]
                    uni, cnt = np.unique(non_zero_id, return_counts=True)
                    cluster_label[np.where(avg_iou >= iou_thr)] = uni[np.where(cnt == cnt.max())].max()

                # print('After Association:', cluster_label)
                cluster_Q[:, 8] = cluster_label
                det_frame[np.where(labels == j), :] = cluster_Q

                #refined_det, refined_mask = final_det_mask(cluster_Q, cluster_i_mask,
                #                                          cluster_i_embed, cluster_center,
                 #                                         time_lag)

                # TODO: obtain final mask and det from the last corner of the temporal window
                refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
                #refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
                #final_mask.append(refined_mask.reshape(28, 28))
                final_det.append(refined_det)
                track_associated = 1
        # [6 6], [6, 6, 6] ... But these are not new track or already associated track
        elif(track_new == 0 and track_associated==0 and cluster_size >= min_cluster_size
             and len(cluster_Q[np.where(cluster_Q[:, 8] == 0)]) == 0):

            #refined_det, refined_mask = final_det_mask(cluster_Q, cluster_i_mask,cluster_i_embed, cluster_center,time_lag)

            # TODO: obtain final mask and det from the last corner of the temporal window
            refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
            #refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
            #final_mask.append(refined_mask.reshape(28, 28))
            final_det.append(refined_det)

    return np.array(final_det), det_frame, ID_ind, cluster_prob_score

