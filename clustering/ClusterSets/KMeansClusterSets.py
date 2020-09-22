from clustering.ClusterSet.KMeansClusterSet import KMeansClusterSet
from clustering.ClusterSets.ClusterSets import ClusterSets


class KMeansClusterSets(ClusterSets):
    """Subclass of ClusterSets, to be used for any range of clusters created by a hierarchical method"""

    def __init__(self, snapshots, cluster_limit_type, cluster_limits):
        """
        Parameters
        __________
        snapshots - a Snapshots object
        cluster_limit_type - this must be 'maxclust'; see KMeansClusterSet
        cluster_limits - the values corresponding to the cluster_limit_type described above; a cluster set will be
            created for every value in this iterable object.
        """

        cluster_sets = [KMeansClusterSet(snapshots, cluster_limit_type, limit) for limit in cluster_limits]
        super().__init__(cluster_sets, snapshots, cluster_limit_type)
