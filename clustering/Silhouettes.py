import numpy as np

from collections import Sequence
from sklearn import metrics


class Silhouettes(Sequence):
    def __init__(self, silhouettes):
        self._silhouettes = silhouettes
        self.averages = np.array([silhouette.average for silhouette in silhouettes])
        self.samples = np.array([silhouette.samples for silhouette in silhouettes])

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Create a 'blank' Silhouettes...
            silhouettes = Silhouettes([])
            # ...and populate its fields with slices from this Silhouettes
            silhouettes._silhouettes = self._silhouettes[key]
            silhouettes.averages = self.averages[key]
            silhouettes.samples = self.samples[key]
            return silhouettes
        else:
            return self._silhouettes[key]

    def __len__(self):
        return len(self._silhouettes)


class Silhouette:
    def __init__(self, distance_data, clusters, metric):
        self.average = metrics.silhouette_score(distance_data, clusters, metric=metric)
        self.samples = metrics.silhouette_samples(distance_data, clusters, metric=metric)
