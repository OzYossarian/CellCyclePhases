import numpy as np
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from drawing.utils import display_name
from collections import Sequence
from clustering.Silhouettes import Silhouettes


class ClusterSets(Sequence):
    def __init__(self, cluster_sets, cluster_data, limit_type):
        self._cluster_sets = cluster_sets
        self.global_data = cluster_data
        self.clusters = np.array([cluster_set.clusters for cluster_set in cluster_sets])
        self.sizes = np.array([cluster_set.size for cluster_set in cluster_sets])
        self.limit_type = limit_type
        self.limits = np.array([cluster_set.limit for cluster_set in cluster_sets])
        self.silhouettes = Silhouettes([cluster_set.silhouette for cluster_set in cluster_sets])

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Create a 'blank' ClusterSets...
            cluster_sets = ClusterSets([], self.limit_type)
            # ...and populate its fields with slices from this ClusterSets
            cluster_sets._cluster_sets = self._cluster_sets[key]
            cluster_sets.clusters = self.clusters[key]
            cluster_sets.sizes = self.sizes[key]
            cluster_sets.limit_type = self.limit_type
            cluster_sets.limits = self.limits[key]
            cluster_sets.silhouettes = self.silhouettes[key]
            return cluster_sets
        else:
            return self._cluster_sets[key]

    def __len__(self):
        return len(self._cluster_sets)


class ClusterSet:
    def __init__(self, clusters, cluster_data, cluster_limit_type, cluster_limit, silhouette):
        self.clusters = clusters
        self.global_data = cluster_data  # ToDo: better name than 'global data'?
        self.size = len(set(clusters))
        self.limit_type = cluster_limit_type
        self.limit = cluster_limit
        self.silhouette = silhouette

    def plot(self, ax=None, y_height=0, title="", number_of_colours=10):
        # ToDo - make plot_cluster_sets call this method
        times = np.array(range(len(self.clusters)))
        ax.scatter(times, times * y_height, c=self.clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
        ax.set_yticks([])
        sb.despine(ax=ax, left=True)
        ax.grid(axis='x')
        ax.set_title(title, weight="bold")

    def plot_dendrogram(self, ax=None, leaf_rotation=90, leaf_font_size=6, title=''):
        if ax is None:
            ax = plt.gca()

        distance_threshold = self.distance_threshold()
        sch.dendrogram(
            self.global_data.linkage,
            leaf_rotation=leaf_rotation,
            leaf_font_size=leaf_font_size,
            color_threshold=distance_threshold,
            above_threshold_color='black',
            ax=ax)

        ax.axhline(y=distance_threshold, c='grey', ls='--', zorder=1)
        ax.set_title(title, weight="bold")
        ax.set_ylabel(display_name(self.limit_type))
        ax.set_xlabel("Time points")

    def distance_threshold(self):
        number_of_observations = self.global_data.linkage.shape[0] + 1
        if self.size >= number_of_observations:
            return 0
        elif self.size <= 1:
            return self.global_data.linkage[-1, 2] * 1.001
        else:
            return self.global_data.linkage[-self.size, 2] * 1.001
