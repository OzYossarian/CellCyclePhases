import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
from clustering.DistanceMatrix import DistanceMatrix


class Snapshots:
    """
    The 'snapshots' of a temporal network are its representation as a sequence of adjacency matrices, one for
    each timepoint. This class holds data related to these snapshots, e.g. the distance matrix representing distances
    between each snapshot. Interestingly this class doesn't actually hold the full snapshots themselves - for our
    purposes it suffices to record the 'flattened' snapshots here, created by turning each snapshot (adjacancy matrix)
    into a vector.
    """

    def __init__(self, flat_snapshots, linkage, distance_matrix, cluster_method, times, true_times):
        """
        Parameters
        __________

        flat_snapshots - a matrix consisting of vectors obtained by flattening each snapshot (adjacency matrix) at each
            time point in the temporal network
        linkage - data representing the hierarchical clustering of these snapshots (if applicable) - e.g. the matrix Z
            returned by SciPy's 'linkage' method.
        distance_matrix - a DistanceMatrix object representing distances between snapshots
        cluster_method - the method used for calculating clusters - e.g. 'ward' or 'k_means'
        times - an array of time points, one for each snapshot, starting at zero
        true_times - an array of time points, one for each snapshot; the actual times used to form the temporal network
        """

        self.flat = flat_snapshots
        self.linkage = linkage
        self.distance_matrix = distance_matrix
        self.cluster_method = cluster_method
        self.times = times
        self.true_times = true_times

    @classmethod
    def from_temporal_network(_class, temporal_network, cluster_method, metric):
        """Create a Snapshots object from a temporal network

        Parameters
        __________
        temporal_network - a TemporalNetwork object (our custom wrapper class, not a teneto.TemporalNetwork object)
        cluster_method - the method used for calculating clusters - e.g. 'ward' or 'k_means'
        metric - the distance metric to be used to calculate distances between snapshots
        """

        linkage, distance_matrix, distance_matrix_condensed = (None, None, None)

        snapshots = temporal_network.get_snapshots()
        # Flatten each each snapshot (i.e. adjacancy matrix) into a vector
        flat_snapshots = snapshots.reshape(temporal_network.T(), -1)

        distance_matrix_full = pairwise_distances(flat_snapshots, metric=metric)
        # A distance matrix like this will be symmetric, so half the entries are redundant; create a condensed version
        # by flattening the upper triangular half of the matrix into a vector
        upper_triangular_indices = np.triu_indices(n=temporal_network.T(), k=1)
        distance_matrix_condensed = distance_matrix_full[upper_triangular_indices]
        distance_matrix = DistanceMatrix(distance_matrix_full, distance_matrix_condensed, metric)

        if cluster_method != 'k_means':
            if len(distance_matrix_condensed) > 0:
                linkage = sch.linkage(distance_matrix_condensed, method=cluster_method)
            else:
                linkage = None

        times, true_times = temporal_network.times, temporal_network.true_times
        return _class(flat_snapshots, linkage, distance_matrix, cluster_method, times, true_times)
