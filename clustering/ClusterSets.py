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
            cluster_sets = ClusterSets([], self.global_data, self.limit_type)
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


class ClusterSet:
    def __init__(self, clusters, cluster_data, cluster_limit_type, cluster_limit, silhouette):
        self.clusters = clusters
        self.global_data = cluster_data  # ToDo: better name than 'global data'?
        self.size = len(set(clusters))
        self.limit_type = cluster_limit_type
        self.limit = cluster_limit
        self.silhouette = silhouette

    def plot(self, ax=None, y_height=0, cmap=cm.tab10, number_of_colors=10):
        times = self.global_data.times
        y = np.ones(len(times)) * y_height
        ax.scatter(times, y, c=self.clusters, cmap=cmap, vmin=1, vmax=number_of_colors)

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

    def plot_silhouette_samples(self, ax=None, title=''):
        if ax is None:
            ax = plt.gca()

        if self.size > 10:
            sb.set_palette("tab20")
        else:
            sb.set_palette("tab10")

        y_lower = 1
        for i, cluster in enumerate(np.unique(self.clusters)):
            # Aggregate the silhouette scores for samples belonging to each cluster, and sort them
            if self.silhouette.samples.size > 0:
                silhouette_values = self.silhouette.samples[self.clusters == cluster]
                silhouette_values.sort()

                silhouette_size = silhouette_values.shape[0]
                y_upper = y_lower + silhouette_size
                y = np.arange(y_lower, y_upper)
                ax.fill_betweenx(y, 0, silhouette_values, facecolor=f"C{i}", edgecolor=f"C{i}", alpha=1)

                # Compute the new y_lower for next plot
                vertical_padding = 1
                y_lower = y_upper + vertical_padding

        ax.set_title(title)
        ax.axvline(x=self.silhouette.average, c='k', ls='--')
        sb.despine()

    def distance_threshold(self):
        number_of_observations = self.global_data.linkage.shape[0] + 1
        if self.size >= number_of_observations:
            return 0
        elif self.size <= 1:
            return self.global_data.linkage[-1, 2] * 1.001
        else:
            return self.global_data.linkage[-self.size, 2] * 1.001
