from sklearn.cluster import KMeans
from clustering.ClusterSets import ClusterSet, ClusterSets
from clustering.Silhouettes import Silhouette


class KMeansClusterSets(ClusterSets):
    def __init__(self, flat_snapshots, cluster_limit_type, cluster_limits, metric):
        cluster_sets = [
            KMeansClusterSet(flat_snapshots, cluster_limit_type, limit, metric)
            for limit in cluster_limits]
        super().__init__(cluster_sets, cluster_limit_type)


class KMeansClusterSet(ClusterSet):
    def __init__(self, flat_snapshots, cluster_limit_type, cluster_limit, metric):
        assert cluster_limit_type == 'maxclust'
        clusters = KMeans(n_clusters=cluster_limit, random_state=None).fit_predict(flat_snapshots) + 1
        silhouette = Silhouette(flat_snapshots, clusters, metric)
        super().__init__(clusters, cluster_limit_type, cluster_limit, silhouette)
