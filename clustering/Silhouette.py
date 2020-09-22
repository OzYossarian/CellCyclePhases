import numpy as np

from sklearn import metrics


class Silhouette:
    """The silhouette of a cluster set is a measure of how 'good' the clustering is"""

    def __init__(self, distance_data, clusters, metric):
        """
        Parameters
        __________
        distance_data - whatever object is being used to represent distances between clusters - e.g. distance matrix.
        clusters - whatever data is being used to represent the clustering - e.g. matrix Z returned by SciPy's
            method 'fcluster'.
        metric - the distance metric to use when calculating silhouette data
        """

        try:
            self.average = metrics.silhouette_score(distance_data, clusters, metric=metric)
            self.samples = metrics.silhouette_samples(distance_data, clusters, metric=metric)
        except ValueError as error:
            # Often the number of clusters is 1, which sklearn does not like.
            print(f'WARNING: unable to compute silhouette for cluster set. Error is: {error}')
            self.average = 0
            self.samples = np.array([])
