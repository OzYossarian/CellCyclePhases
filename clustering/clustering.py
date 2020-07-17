import numpy as np
import scipy.cluster.hierarchy as sch

from sklearn.metrics.pairwise import cosine_similarity


def linkage(temporal_network, distance_type, method):
    _, distance_matrix_condensed = compute_snapshot_distances(temporal_network, distance_type)
    return sch.linkage(distance_matrix_condensed, method=method)


def compute_snapshot_distances(temporal_network, distance_type='euclidean'):
    """
    Compute pairwise distance between snapshots

    parameters
    ----------
    - dist : string ('cosinesim', 'euclidean', 'euclidean_flat')

    returns
    -------

    """

    T = temporal_network.T
    dist_mat = np.zeros((T, T))

    snapshots = temporal_network.df_to_array()
    snapshots = np.swapaxes(snapshots, 0, 2)  # put time as zeroth axis
    snapshot_flat = snapshots.reshape(T, -1)  # each matrix is flattened, represented as a vector

    if distance_type == 'cosinesim':
        dist_mat = 1 - cosine_similarity(snapshot_flat, snapshot_flat)
        np.fill_diagonal(dist_mat, 0)  # otherwise, 1e-15 but negative values, cause problems later

    else:
        if distance_type == "euclidean_flat":
            snapshots = snapshot_flat
        for j in range(T):
            for i in range(j):  # fill upper triangle only
                dist_mat[i, j] = np.linalg.norm(snapshots[i] - snapshots[j])  # Eucledian distance

        dist_mat = dist_mat + dist_mat.T

    # extract condensed distance matrix needed for the clustering
    id_u = np.triu_indices(n=T, k=1)  # indices of upper triangle elements
    dist_mat_condensed = dist_mat[id_u]

    return dist_mat, dist_mat_condensed
