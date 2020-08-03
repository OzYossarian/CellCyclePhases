import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
from clustering.DistanceMatrix import DistanceMatrix


class Snapshots:
    def __init__(self, flat_snapshots, linkage, distance_matrix, cluster_method, times):
        self.flat = flat_snapshots
        self.linkage = linkage
        self.distance_matrix = distance_matrix
        self.cluster_method = cluster_method
        self.times = times

    @classmethod
    def from_temporal_network(_class, temporal_network, method, metric):
        linkage, distance_matrix, distance_matrix_condensed = (None, None, None)

        snapshots = temporal_network.get_snapshots()
        times = range(snapshots.shape[0])
        flat_snapshots = snapshots.reshape(temporal_network.T, -1)

        distance_matrix_full = pairwise_distances(flat_snapshots, metric=metric)
        upper_triangular_indices = np.triu_indices(n=temporal_network.T, k=1)
        distance_matrix_condensed = distance_matrix_full[upper_triangular_indices]
        distance_matrix = DistanceMatrix(distance_matrix_full, distance_matrix_condensed, metric)

        if method != 'k_means':
            linkage = sch.linkage(distance_matrix_condensed, method=method)

        return _class(flat_snapshots, linkage, distance_matrix, method, times)
