from sklearn.cluster import KMeans
from clustering.ClusterSets import ClusterSet, ClusterSets
from clustering.Silhouettes import Silhouette


class KMeansClusterSets(ClusterSets):
    def __init__(self, cluster_data, cluster_limit_type, cluster_limits):
        cluster_sets = [
            KMeansClusterSet(cluster_data, cluster_limit_type, limit)
            for limit in cluster_limits]
        super().__init__(cluster_sets, cluster_data, cluster_limit_type)


class KMeansClusterSet(ClusterSet):
    def __init__(self, cluster_data, cluster_limit_type, cluster_limit):
        assert cluster_limit_type == 'maxclust'
        k_means = KMeans(n_clusters=cluster_limit, random_state=None)
        clusters = k_means.fit_predict(cluster_data.flat_snapshots) + 1
        silhouette = Silhouette(cluster_data.flat_snapshots, clusters, cluster_data.metric)
        super().__init__(clusters, cluster_data, cluster_limit_type, cluster_limit, silhouette)
