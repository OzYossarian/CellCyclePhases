import scipy.cluster.hierarchy as sch

from clustering.ClusterSet.ClusterSet import ClusterSet
from clustering.Silhouette import Silhouette


class HierarchicalClusterSet(ClusterSet):
    def __init__(self, cluster_data, cluster_limit_type, cluster_limit):
        clusters = sch.fcluster(cluster_data.linkage, cluster_limit, criterion=cluster_limit_type)
        silhouette = Silhouette(cluster_data.distance_matrix, clusters, metric='precomputed')
        super().__init__(clusters, cluster_data, cluster_limit_type, cluster_limit, silhouette)