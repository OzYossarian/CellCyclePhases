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
    def via_hierarchical_clustering(
            _class,
            linkage,
            distance_matrix,
            cluster_limit_type,
            cluster_limits,
            number_of_time_points):

        # Define methods for getting hierarchical clusters, as well as silhouette average and sample
        def get_clusters(cluster_limit):
            return sch.fcluster(linkage, cluster_limit, criterion=cluster_limit_type)

        def get_silhouettes(clusters):
            average = metrics.silhouette_score(distance_matrix, clusters, metric='precomputed')
            samples = metrics.silhouette_samples(distance_matrix, clusters, metric='precomputed')
            return average, samples

        return _class.via(get_clusters, get_silhouettes, cluster_limits, number_of_time_points)

    @classmethod
    def via_k_means_clustering(
            _class,
            flat_snapshots,
            cluster_limit_type,
            cluster_limits,
            metric,
            number_of_time_points):

        # Define methods for getting K-means clusters, as well as silhouette average and sample
        assert cluster_limit_type == 'maxclust'

        def get_clusters(max_clusters):
            # ToDo: Could combine this method with the place in the notebook where we create a single set of clusters?
            k_means = KMeans(n_clusters=max_clusters, random_state=None).fit_predict(flat_snapshots)
            # Add 1, so that clusters are 1-indexed; this makes plotting easier.
            return k_means + 1

        def get_silhouettes(clusters):
            average = metrics.silhouette_score(flat_snapshots, clusters, metric=metric)
            samples = metrics.silhouette_samples(flat_snapshots, clusters, metric=metric)
            return average, samples

        return _class.via(get_clusters, get_silhouettes, cluster_limits, number_of_time_points)

    @classmethod
    def via(_class, get_clusters, get_silhouettes, cluster_limits, number_of_time_points):
        size = len(cluster_limits)

        # One set of flat clusters per number in max_cluster_range
        clusters = np.zeros((size, number_of_time_points))
        numbers_of_clusters = np.zeros(size)

        # One set of silhouette samples per number in max_cluster_range
        silhouette_samples = np.zeros((size, number_of_time_points))
        average_silhouettes = np.zeros(size)

        for i, cluster_limit in enumerate(cluster_limits):
            clusters[i] = get_clusters(cluster_limit)
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
