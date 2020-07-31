import numpy as np

from sklearn import metrics


class Silhouette:
    def __init__(self, distance_data, clusters, metric):
        try:
            self.average = metrics.silhouette_score(distance_data, clusters, metric=metric)
            self.samples = metrics.silhouette_samples(distance_data, clusters, metric=metric)
        except ValueError as error:
            # Often the number of clusters is 1, which sklearn does not like.
            print(f'WARNING: unable to compute silhouette for cluster set. Error is: {error}')
            self.average = 0
            self.samples = np.array([])
