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
    distance_matrix = np.zeros((T, T))

    snapshots = temporal_network.df_to_array()
    snapshots = np.swapaxes(snapshots, 0, 2)  # put time as zeroth axis
    snapshot_flat = snapshots.reshape(T, -1)  # each matrix is flattened, represented as a vector

    if distance_type == 'cosinesim':
        distance_matrix = 1 - cosine_similarity(snapshot_flat, snapshot_flat)
        np.fill_diagonal(distance_matrix, 0)  # otherwise, 1e-15 but negative values, cause problems later

    else:
        if distance_type == "euclidean_flat":
            snapshots = snapshot_flat
        for j in range(T):
            for i in range(j):  # fill upper triangle only
                distance_matrix[i, j] = np.linalg.norm(snapshots[i] - snapshots[j])  # Eucledian distance

        distance_matrix = distance_matrix + distance_matrix.T

    # extract condensed distance matrix needed for the clustering
    upper_triangular_indices = np.triu_indices(n=T, k=1)  # indices of upper triangle elements
    distance_matrix_condensed = distance_matrix[upper_triangular_indices]

    return distance_matrix, distance_matrix_condensed
