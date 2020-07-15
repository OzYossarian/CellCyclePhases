import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def compute_snapshot_distances(tnet, dist='eucledian'):
    snapshots = tnet.df_to_array()
    # put time as zeroth axis
    snapshots = np.swapaxes(snapshots, 0, 2)

    T = tnet.T
    dist_mat = np.zeros((T, T))

    for i in range(T):
        for j in range(i):
            # Eucledian distance
            dist_mat[j, i] = np.linalg.norm(snapshots[i] - snapshots[j])

            # extract condensed distance matrix need for the clustering
    # id_l = np.tril_indices(n=T, k=-1) # indices of lower triangle elements
    id_u = np.triu_indices(n=T, k=1)  # indices of lower triangle elements

    # dist_mat_condensed = dist_mat[id_l]
    dist_mat_condensed = dist_mat[id_u]

    return dist_mat, dist_mat_condensed

