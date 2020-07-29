import numpy as np
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from drawing.utils import display_name


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
