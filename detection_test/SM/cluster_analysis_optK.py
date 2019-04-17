from __future__ import division
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from opt_K_elbow import Elbow_opt_K
################################################################
#
#  Use only when the number of maximum instances from the baseline model is >=2
#
#
################################################################
def optimal_k_value(latent_feature,k_base):
    if(len(latent_feature)>10):
        range_n_clusters = np.arange(2,k_base+4)
    else:
        range_n_clusters = np.arange(2, len(latent_feature))

    silhouette_avg = {}
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        #ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        #ax1.set_ylim([0, len(latent_feature) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(latent_feature)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        score = silhouette_score(latent_feature, cluster_labels)
        silhouette_avg.setdefault('silhouette_avg_score', [])
        silhouette_avg.setdefault('k_value', [])
        silhouette_avg['silhouette_avg_score'].append(score)
        silhouette_avg['k_value'].append(n_clusters)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", score)

        # Compute the silhouette scores for each sample
        #sample_silhouette_values = silhouette_samples(latent_feature, cluster_labels)

        '''
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        '''
    silhouette_avg_score = np.array(silhouette_avg['silhouette_avg_score'])
    # when multiple silhouette average score becomes similar ????
    best_avg_score = np.around(silhouette_avg_score, decimals=1)

    k_index = np.array(silhouette_avg['k_value'])
    opt_score = silhouette_avg_score[np.where(silhouette_avg_score==silhouette_avg_score.max())]
    opt_k = k_index[np.where(silhouette_avg_score==silhouette_avg_score.max())][0]

    # out of scope from silhouette score
    '''
    if(opt_score < 0.5 and k_base==2):
       opt_k = np.array(k_base - 1)
    '''
    if(k_base - opt_k)>2:
      #call elbow method
      opt_k = Elbow_opt_K(latent_feature,k_base)
    return opt_k