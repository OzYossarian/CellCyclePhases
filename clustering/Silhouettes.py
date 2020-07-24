import numpy as np
import scipy.cluster.hierarchy as sch

from collections.abc import Sequence
from sklearn import metrics
from sklearn.cluster import KMeans

from clustering.Silhouette import Silhouette


class Silhouettes(Sequence):
    def __init__(self, clusters, numbers_of_clusters, samples, averages):
        self.clusters = clusters
        self.numbers_of_clusters = numbers_of_clusters
        self.samples = samples
        self.averages = averages
        super().__init__()

    @classmethod
    def via_flat_clusters(_class, linked_clusters, distance_matrix, max_cluster_range, metric, number_of_time_points):
        # Define methods for getting flat clusters, as well as silhouette average and sample
        def get_clusters(max_clusters):
            return sch.fcluster(linked_clusters, max_clusters, criterion='maxclust')

        def get_silhouettes(clusters):
            average = metrics.silhouette_score(distance_matrix, clusters, metric=metric)
            samples = metrics.silhouette_samples(distance_matrix, clusters, metric=metric)
            return average, samples

        return _class.via(get_clusters, get_silhouettes, max_cluster_range, number_of_time_points)

    @classmethod
    def via_k_means(_class, flat_snapshot, max_cluster_range, metric, number_of_time_points):
        # Define methods for getting K-means clusters, as well as silhouette average and sample
        def get_clusters(max_clusters):
            return KMeans(n_clusters=max_clusters, random_state=None).fit_predict(flat_snapshot)

        def get_silhouettes(clusters):
            average = metrics.silhouette_score(flat_snapshot, clusters, metric=metric)
            samples = metrics.silhouette_samples(flat_snapshot, clusters, metric=metric)
            return average, samples

        return _class.via(get_clusters, get_silhouettes, max_cluster_range, number_of_time_points)

    @classmethod
    def via(_class, get_clusters, get_silhouettes, max_cluster_range, number_of_time_points):
        max_cluster_range_length = len(max_cluster_range)

        # One set of flat clusters per number in max_cluster_range
        clusters = np.zeros((max_cluster_range_length, number_of_time_points))
        numbers_of_clusters = np.zeros(max_cluster_range_length)

        # One set of silhouette samples per number in max_cluster_range
        silhouette_samples = np.zeros((max_cluster_range_length, number_of_time_points))
        average_silhouettes = np.zeros(max_cluster_range_length)

        for i, max_clusters in enumerate(max_cluster_range):
            clusters[i] = get_clusters(max_clusters)
            numbers_of_clusters[i] = len(set(clusters[i]))

            if numbers_of_clusters[i] > 1:
                average_silhouettes[i], silhouette_samples[i] = get_silhouettes(clusters[i])

        return _class(clusters, numbers_of_clusters, silhouette_samples, average_silhouettes)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Silhouettes(self.clusters[key], self.numbers_of_clusters[key], self.samples[key], self.averages[key])
        else:
            return Silhouette(self.clusters[key], self.numbers_of_clusters[key], self.samples[key], self.averages[key])

    def __len__(self):
        return self.clusters.shape[0]
