import scipy.cluster.hierarchy as sch

from clustering.ClusterSet.ClusterSet import ClusterSet
from clustering.Silhouette import Silhouette


class HierarchicalClusterSet(ClusterSet):
    def __init__(self, snapshots, cluster_limit_type, cluster_limit):
        clusters = sch.fcluster(snapshots.linkage, cluster_limit, criterion=cluster_limit_type)
        silhouette = Silhouette(snapshots.distance_matrix.full, clusters, metric='precomputed')
        super().__init__(clusters, snapshots, cluster_limit_type, cluster_limit, silhouette)