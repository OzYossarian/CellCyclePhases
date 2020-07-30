import numpy as np
import matplotlib.pyplot as plt

from drawing.utils import display_name
from collections import Sequence
from clustering.Silhouettes import Silhouettes


class ClusterSets(Sequence):
    def __init__(self, cluster_sets, snapshots, limit_type):
        self._cluster_sets = cluster_sets
        self.snapshots = snapshots
        self.clusters = np.array([cluster_set.clusters for cluster_set in cluster_sets])
        self.sizes = np.array([cluster_set.size for cluster_set in cluster_sets])
        self.limit_type = limit_type
        self.limits = np.array([cluster_set.limit for cluster_set in cluster_sets])
        self.silhouettes = Silhouettes([cluster_set.silhouette for cluster_set in cluster_sets])

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Create a 'blank' ClusterSets...
            cluster_sets = ClusterSets([], self.snapshots, self.limit_type)
            # ...and populate its fields with slices from this ClusterSets
            cluster_sets._cluster_sets = self._cluster_sets[key]
            cluster_sets.clusters = self.clusters[key]
            cluster_sets.sizes = self.sizes[key]
            cluster_sets.limits = self.limits[key]
            cluster_sets.silhouettes = self.silhouettes[key]
            return cluster_sets
        else:
            return self._cluster_sets[key]

    def __len__(self):
        return len(self._cluster_sets)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        for cluster_set in self._cluster_sets:
            (cmap, number_of_colors) = (plt.cm.tab20, 20) if cluster_set.size > 10 else (plt.cm.tab10, 10)
            cluster_set.plot(ax=ax, y_height=cluster_set.limit, cmap=cmap, number_of_colors=number_of_colors)

        # ToDo - this stuff shouldn't be in this 'plot' method. Should have separate methods for formatting.
        # Leave some space at the bottom in which to plot phases later
        limits_size = (self.limits[-1] - self.limits[0])
        ylim_bottom = self.limits[0] - limits_size * 0.25
        ylim_top = self.limits[-1] + limits_size * 0.08
        ax.set_ylim([ylim_bottom, ylim_top])

        ax.set_xlabel("Times (min)")
        ax.set_axisbelow(True)

    def plot_with_average_silhouettes(self, axs):
        self.plot(ax=axs[0])
        self.plot_average_silhouettes(ax=axs[1])
        self.plot_sizes(ax=axs[2])

        axs[1].yaxis.set_tick_params(labelleft=True)
        axs[2].yaxis.set_tick_params(labelleft=True)

        axs[0].set_ylabel(display_name(self.limit_type))
        plt.subplots_adjust(wspace=0.4, top=0.8)

    def plot_average_silhouettes(self, ax):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.silhouettes.averages, self.limits, 'ko-')
        ax.set_xlabel("Average silhouette")
        ax.set_xlim((-0.1, 1.1))

    def plot_sizes(self, ax):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.sizes, self.limits, 'ko-')
        ax.set_xlabel("Actual # clusters")

    def plot_silhouette_samples(self, axs):
        for i, cluster_set in enumerate(self._cluster_sets):
            ax = axs.flatten()[i]
            title = f'{display_name(cluster_set.limit_type)} = {cluster_set.limit}'
            subtitle = f'({int(cluster_set.size)} clusters)'
            cluster_set.plot_silhouette_samples(ax=ax, title=f'{title}\n{subtitle}')
