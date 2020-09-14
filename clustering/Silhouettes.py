import numpy as np

from collections import Sequence


class Silhouettes(Sequence):
    """
    Class representing silhouettes for a range of cluster sets, as opposed to a single cluster set.
    As with the ClusterSets class, when plotting it's useful to have a class dedicated to representing a range
    of silhouettes, rather than just using a Python list of Silhouette objects.
    """
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
