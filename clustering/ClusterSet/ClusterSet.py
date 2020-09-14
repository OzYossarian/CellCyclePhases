import numpy as np
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


class ClusterSet:
    """
    Class representing a set of clusters

    limit_type refers to the method used to decide when to stop clustering when creating this cluster set.
    e.g. a cluster set can be created by clustering until a partiuclar number of clusters has been reached, or
    until every cluster is at least a certain distance away from each other.
    """
    def __init__(self, clusters, snapshots, cluster_limit_type, cluster_limit, silhouette):
        self.clusters = clusters
        self.snapshots = snapshots
        self.size = len(set(clusters))
        self.limit_type = cluster_limit_type
        self.limit = cluster_limit
        self.silhouette = silhouette

    def plot(self, ax=None, y_height=0, cmap=cm.get_cmap('tab10'), number_of_colors=10):
        times = self.snapshots.times
        y = np.ones(len(times)) * y_height
        ax.scatter(times, y, c=self.clusters, cmap=cmap, vmin=1, vmax=number_of_colors)

    def plot_dendrogram(self, ax=None, leaf_rotation=90, leaf_font_size=6):
        if ax is None:
            ax = plt.gca()

        # Calculate the distance threshold at which clusters stop, so that below this threshold we plot the
        # dendrogram in colour, while above it we plot in black.
        distance_threshold = self.distance_threshold()

        sch.dendrogram(
            self.snapshots.linkage,
            leaf_rotation=leaf_rotation,
            leaf_font_size=leaf_font_size,
            color_threshold=distance_threshold,
            above_threshold_color='black',
            ax=ax)
        ax.axhline(y=distance_threshold, c='grey', ls='--', zorder=1)

    def plot_silhouette_samples(self, ax=None):
        if ax is None:
            ax = plt.gca()

        # If there are more than 10 clusters in this cluster set, we'll need to use more colours in our plot.
        sb.set_palette("tab20" if self.size > 10 else "tab10")

        if self.silhouette.samples.size > 0:
            y_lower = 0
            for i, cluster in enumerate(np.unique(self.clusters)):
                # Aggregate the silhouette scores for samples belonging to each cluster, and sort them
                silhouette_values = self.silhouette.samples[self.clusters == cluster]
                silhouette_values.sort()
                silhouette_size = silhouette_values.shape[0]

                # Calculate height of this cluster
                y_upper = y_lower + silhouette_size
                y = np.arange(y_lower, y_upper)
                ax.fill_betweenx(y, 0, silhouette_values, facecolor=f"C{i}", edgecolor=f"C{i}", alpha=1)

                # Compute the new y_lower for next cluster
                vertical_padding = 0
                y_lower = y_upper + vertical_padding

        ax.axvline(x=self.silhouette.average, c='k', ls='--')

    def distance_threshold(self):
        number_of_observations = self.snapshots.linkage.shape[0] + 1
        if self.size >= number_of_observations:
            return 0
        elif self.size <= 1:
            return self.snapshots.linkage[-1, 2] * 1.001
        else:
            return self.snapshots.linkage[-self.size, 2] * 1.001
