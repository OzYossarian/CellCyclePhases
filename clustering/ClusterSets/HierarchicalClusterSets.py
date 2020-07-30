from clustering.ClusterSet.HierarchicalClusterSet import HierarchicalClusterSet
from clustering.ClusterSets.ClusterSets import ClusterSets


class HierarchicalClusterSets(ClusterSets):
    def __init__(self, snapshots, cluster_limit_type, cluster_limits):
        cluster_sets = [HierarchicalClusterSet(snapshots, cluster_limit_type, limit) for limit in cluster_limits]
        super().__init__(cluster_sets, snapshots, cluster_limit_type)
