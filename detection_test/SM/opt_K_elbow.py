from __future__ import division
from scipy.spatial.distance import cdist
import kneed
import numpy as np
from sklearn.cluster import KMeans

def Elbow_opt_K(latent_feature,max_instances):
    k_eval = {}
    for k in range(1, max_instances + 5):
        km = KMeans(n_clusters=k, max_iter=500).fit(latent_feature)
        # k_eval[k] =km.inertia_ # sum of distanceof samples to their closest cluster center
        k_eval[k] = km.inertia_#np.sum(np.min(cdist(latent_feature, km.cluster_centers_, 'euclidean'), axis=1)) / latent_feature.shape[0]
    # plt.figure()
    # plt.plot(list(k_eval.keys()),list(k_eval.values()))
    # plt.xlabel('k value')
    # plt.ylabel('SSE')
    # plt.savefig('/media/siddique/Data/CLASP2018/cluster_result/6A/cam11/final_model/overlaid_mask/' +'elbow' + names[-10:],dpi=dpi)
    # plt.show()
    # finad optimal number of cluster from SSE score
    k_opt = kneed.KneeLocator(list(k_eval.keys()), list(k_eval.values()), curve='convex',direction='decreasing')
    max_instances_at_theta = k_opt.elbow_x
    return max_instances_at_theta