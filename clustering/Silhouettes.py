import numpy as np
import scipy.cluster.hierarchy as sch

from collections.abc import Sequence
from sklearn import metrics

from clustering.Silhouette import Silhouette


class Silhouettes(Sequence):
    def __init__(self, clusters, numbers_of_clusters, samples, averages):
        self.clusters = clusters
        self.numbers_of_clusters = numbers_of_clusters
        self.samples = samples
        self.averages = averages
        super().__init__()

    @classmethod
    def for_max_cluster_range(_class, linked_clusters, distance_matrix, max_cluster_range, number_of_time_points):
        max_cluster_range_length = len(max_cluster_range)

        # One set of flat clusters per number in max_cluster_range
        clusters = np.zeros((max_cluster_range_length, number_of_time_points))
        numbers_of_clusters = np.zeros(max_cluster_range_length)

        # One set of silhouette samples per number in max_cluster_range
        silhouette_samples = np.zeros((max_cluster_range_length, number_of_time_points))
        average_silhouettes = np.zeros(max_cluster_range_length)

        for i, max_clusters in enumerate(max_cluster_range):
            clusters[i] = sch.fcluster(linked_clusters, max_clusters, criterion='maxclust')
            numbers_of_clusters[i] = len(set(clusters[i]))

            if numbers_of_clusters[i] > 1:
                average_silhouettes[i] = metrics.silhouette_score(distance_matrix, clusters[i], metric="precomputed")
                silhouette_samples[i] = metrics.silhouette_samples(distance_matrix, clusters[i], metric="precomputed")

        return _class(clusters, numbers_of_clusters, silhouette_samples, average_silhouettes)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Silhouettes(self.clusters[key], self.numbers_of_clusters[key], self.samples[key], self.averages[key])
        else:
            return Silhouette(self.clusters[key], self.numbers_of_clusters[key], self.samples[key], self.averages[key])

    def __len__(self):
        return self.clusters.shape[0]
