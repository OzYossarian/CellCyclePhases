from sklearn.cluster import KMeans

from clustering.ClusterSet.ClusterSet import ClusterSet
from clustering.Silhouette import Silhouette


class KMeansClusterSet(ClusterSet):
    """Subclass of ClusterSet specifically for clusters created using the k-means method."""

    def __init__(self, snapshots, cluster_limit_type, cluster_limit):
        """
        See base class for parameter info
        """

        # k-means clustering is only applicable for cluster_limit_type of 'maxclust'
        assert cluster_limit_type == 'maxclust'
        k_means = KMeans(n_clusters=cluster_limit, random_state=None)
        clusters = k_means.fit_predict(snapshots.flat) + 1
        silhouette = Silhouette(snapshots.flat, clusters, snapshots.distance_matrix.metric)
        super().__init__(clusters, snapshots, cluster_limit_type, cluster_limit, silhouette)