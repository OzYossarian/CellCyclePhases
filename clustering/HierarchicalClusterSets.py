import scipy.cluster.hierarchy as sch

from clustering.ClusterSets import ClusterSet, ClusterSets
from clustering.Silhouettes import Silhouette


class HierarchicalClusterSets(ClusterSets):
    def __init__(self, linkage, distance_matrix, cluster_limit_type, cluster_limits):
        cluster_sets = [
            HierarchicalClusterSet(linkage, distance_matrix, cluster_limit_type, limit)
            for limit in cluster_limits]
        super().__init__(cluster_sets, cluster_limit_type)


class HierarchicalClusterSet(ClusterSet):
    def __init__(self, linkage, distance_matrix, cluster_limit_type, cluster_limit):
        clusters = sch.fcluster(linkage, cluster_limit, criterion=cluster_limit_type)
        silhouette = Silhouette(distance_matrix, clusters, metric='precomputed')
        super().__init__(clusters, cluster_limit_type, cluster_limit, silhouette)
