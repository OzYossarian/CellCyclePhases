import numpy as np
import matplotlib.pyplot as plt

from collections import Sequence
from clustering.Silhouettes import Silhouettes
from utils import drawing
from utils.plotting import plot_phases, plot_events


class ClusterSets(Sequence):
    """Class representing a range of cluster sets

    e.g. for a range of cluster sets created by stopping clustering after 2 clusters have formed, then 3 clusters,
    then 4, ..., etc. In order to plot data across a range of cluster sets, it is useful to have a dedicated class,
    rather than (e.g.) just using a Python list of ClusterSet objects

    __getitem__ and __len__ are the the two methods of the Sequence base class that we must override
    """

    def __init__(self, cluster_sets, snapshots, limit_type):
        """
        Parameters
        __________
        cluster_sets - an iterable of ClusterSet objects
        snapshots - a Snapshots object
        limit_type - the method that was used to determine when to stop clustering when creating these cluster
            sets. e.g. A cluster set can be created by clustering until a particular number of clusters has been
            reached ('maxclust'), or until every cluster is at least a certain distance away from each other
            ('distance').
        """

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
        """Plot these cluster sets as a scatter graph

        Parameters
        __________
        ax - the matplotlib axes on which to plot
        """

        if ax is None:
            ax = plt.gca()
        for cluster_set in self._cluster_sets:
            (cmap, number_of_colors) = ('tab20', 20) if cluster_set.size > 10 else ('tab10', 10)
            cluster_set.plot(
                ax=ax, y_height=cluster_set.limit, cmap=plt.cm.get_cmap(cmap), number_of_colors=number_of_colors)

    def plot_with_average_silhouettes(self, axs):
        """Plot these cluster sets as a scatter graph, along with the average silhouettes and cluster set sizes

        Parameters
        __________
        axs - the matplotlib axes on which to plot; should be an indexable object with at least three items
        """

        self.plot(ax=axs[0])
        self.plot_average_silhouettes(ax=axs[1])
        self.plot_sizes(ax=axs[2])

    def plot_and_format_with_average_silhouettes(self, axs, events, phases, time_ticks=None):
        """Plot and format these cluster sets as a scatter graph, along with the average silhouettes and cluster set
        sizes

        Our pattern generally has been to leave all formatting in the jupyter notebooks, but this method is used
        by several different notebooks, so it makes sense to put it somewhere common.

        Parameters
        __________
        axs - the matplotlib axes on which to plot; should be an indexable object with at least three items
        events - any events that should be plotted on the scatter graph
        phases - any phases that should be plotted on the scatter graph
        time_ticks - the ticks that should be displayed along the x-axis (time axis)
        """

        (ax1, ax2, ax3) = (axs[0], axs[1], axs[2])

        # Plot
        ax3.tick_params(labelleft=True, labelbottom=True)
        self.plot_with_average_silhouettes((ax1, ax2, ax3))
        drawing.adjust_margin(ax1, bottom=(0.15 if phases else 0))
        plot_events(events, ax=ax1)
        plot_phases(phases, ax=ax1, y_pos=0.04, ymax=0.1)

        # Format
        ax1.set_xlabel("Time")
        ax1.set_ylabel(drawing.display_name(self.limit_type))
        ax1.tick_params(labelbottom=True)
        if time_ticks:
            ax1.set_xticks(time_ticks)

        ax2.set_xlabel("Average silhouette")
        ax2.set_xlim((0, 1))
        ax2.tick_params(labelleft=True, labelbottom=True)

        ax3.set_xlabel("Actual # clusters")

    def plot_average_silhouettes(self, ax):
        """Plot the average silhouettes across this range of cluster sets

        Parameters
        __________
        ax - the matplotlib axes on which to plot
        """

        if ax is None:
            ax = plt.gca()
        ax.plot(self.silhouettes.averages, self.limits, 'ko-')

    def plot_sizes(self, ax):
        """Plot the average cluster set sizes across this range of cluster sets

        Parameters
        __________
        ax - the matplotlib axes on which to plot
        """

        if ax is None:
            ax = plt.gca()
        ax.plot(self.sizes, self.limits, 'ko-')

    def plot_silhouette_samples(self, axs):
        """Plot the average silhouettes across this range of cluster sets

        Parameters
        __________
        axs - the matplotlib axes on which to plot; should be an iterable object with at least as many items as there
            are cluster sets in this class.
        """

        for cluster_set, ax in zip(self._cluster_sets, axs.flatten()):
            cluster_set.plot_silhouette_samples(ax=ax)
