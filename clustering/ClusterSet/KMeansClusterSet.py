from sklearn.cluster import KMeans

from clustering.ClusterSet.ClusterSet import ClusterSet
from clustering.Silhouette import Silhouette


class KMeansClusterSet(ClusterSet):
    def __init__(self, snapshots, cluster_limit_type, cluster_limit):
        assert cluster_limit_type == 'maxclust'
        k_means = KMeans(n_clusters=cluster_limit, random_state=None)
        clusters = k_means.fit_predict(snapshots.flat) + 1
        silhouette = Silhouette(snapshots.flat, clusters, snapshots.metric)
        super().__init__(clusters, snapshots, cluster_limit_type, cluster_limit, silhouette)