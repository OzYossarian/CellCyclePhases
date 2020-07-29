import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances


class ClusterData:
    def __init__(self, flat_snapshots, linkage, distance_matrix, distance_matrix_condensed, method, metric, times):
        self.flat_snapshots = flat_snapshots
        self.linkage = linkage
        self.distance_matrix = distance_matrix
        self.distance_matrix_condensed = distance_matrix_condensed
        self.method = method
        self.metric = metric
        self.times = times

    @classmethod
    def from_temporal_network(_class, temporal_network, method, metric):
        linkage, distance_matrix, distance_matrix_condensed = (None, None, None)

        snapshots = temporal_network.df_to_array()
        # Put time as zeroth axis and flatten each matrix to a vector
        snapshots = np.swapaxes(snapshots, 0, 2)
        times = range(snapshots.shape[0])
        flat_snapshots = snapshots.reshape(temporal_network.T, -1)

        if method != 'k_means':
            distance_matrix = pairwise_distances(flat_snapshots, metric=metric)
            upper_triangular_indices = np.triu_indices(n=temporal_network.T, k=1)
            distance_matrix_condensed = distance_matrix[upper_triangular_indices]
            linkage = sch.linkage(distance_matrix_condensed, method=method)

        return _class(flat_snapshots, linkage, distance_matrix, distance_matrix_condensed, method, metric, times)
