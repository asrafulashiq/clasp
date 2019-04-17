from __future__ import division
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from opt_K_elbow import Elbow_opt_K
import math

################################################################
#
#  Use only when the number of maximum instances from the baseline model is >=2
#
#
################################################################

def optimal_k_value(latent_feature, k_base,pax_boxs,time_lag):
    range_n_clusters = np.arange(3, k_base+5)

    silhouette_avg = {}
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        # ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        # ax1.set_ylim([0, len(latent_feature) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster_center, labels, inertia = k_means(latent_feature, n_clusters=n_clusters)
        #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        #cluster_labels = clusterer.fit_predict(latent_feature)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        score = silhouette_score(latent_feature, labels)
        silhouette_avg.setdefault('silhouette_avg_score', [])
        silhouette_avg.setdefault('k_value', [])
        silhouette_avg.setdefault('cluster_labels',[])
        silhouette_avg['silhouette_avg_score'].append(score)
        silhouette_avg['k_value'].append(n_clusters)
        silhouette_avg['cluster_labels'].append(labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", score)

    silhouette_avg_score = np.array(silhouette_avg['silhouette_avg_score'])
    # when multiple silhouette average score becomes similar ????
    #best_avg_score = np.around(silhouette_avg_score, decimals=2)
    best_avg_score = silhouette_avg_score

    k_index = np.array(silhouette_avg['k_value'])
    score_sorted = np.sort(best_avg_score)
    labels_k = np.array(silhouette_avg['cluster_labels'])
    best_ks_labels = labels_k[np.where(silhouette_avg_score >= score_sorted[-3:].min()), :][0]
    best_ks = k_index[np.where(silhouette_avg_score >= score_sorted[-3:].min())]
    # measure uniqueness of frames for each best k
    q = []
    for i in range(len(best_ks)):
        fr_pattern_c = [pax_boxs[:,0][np.where(best_ks_labels[i]==c)] for c in np.unique(best_ks_labels[i])]
        cluster_rank = [j for j in range(len(fr_pattern_c)) if len(fr_pattern_c[j])<=time_lag]
        q.append(len(cluster_rank))

    #opt_score = silhouette_avg_score[np.max(np.where(best_avg_score == best_avg_score.max()))]
    #opt_k = k_index[np.max(np.where(best_avg_score == best_avg_score.max()))]
    q = np.array(q)
    opt_k = best_ks[np.where(q==q.max())][0]

    return opt_k