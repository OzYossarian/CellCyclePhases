from clustering.ClusterSet.KMeansClusterSet import KMeansClusterSet
from clustering.ClusterSets.ClusterSets import ClusterSets


class KMeansClusterSets(ClusterSets):
    def __init__(self, snapshots, cluster_limit_type, cluster_limits):
        cluster_sets = [KMeansClusterSet(snapshots, cluster_limit_type, limit) for limit in cluster_limits]
        super().__init__(cluster_sets, snapshots, cluster_limit_type)
