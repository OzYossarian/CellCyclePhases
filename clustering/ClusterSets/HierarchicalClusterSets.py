from clustering.ClusterSet.HierarchicalClusterSet import HierarchicalClusterSet
from clustering.ClusterSets.ClusterSets import ClusterSets


class HierarchicalClusterSets(ClusterSets):
    def __init__(self, cluster_data, cluster_limit_type, cluster_limits):
        cluster_sets = [
            HierarchicalClusterSet(cluster_data, cluster_limit_type, limit)
            for limit in cluster_limits]
        super().__init__(cluster_sets, cluster_data, cluster_limit_type)
