import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
import seaborn as sb


def plot_dendrogram_from_clusters(linkage, cluster_set_size, ax=None, leaf_rotation=90, leaf_font_size=6, title=''):
    if ax is None:
        ax = plt.gca()

    distance_threshold = get_distance_threshold(linkage, cluster_set_size)
    sch.dendrogram(
        linkage,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=distance_threshold,
        above_threshold_color='black',
        ax=ax)

    ax.axhline(y=distance_threshold, c='grey', ls='--', zorder=1)
    ax.set_title(title, weight="bold")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Time points")


def plot_scatter_of_clusters(clusters, times, ax=None, title="", number_of_colours=10):
    ax.scatter(times, times * 0, c=clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
    ax.set_yticks([])
    sb.despine(ax=ax, left=True)
    ax.grid(axis='x')
    ax.set_title(title, weight="bold")


def get_distance_threshold(linkage, cluster_set_size):
    number_of_observations = linkage.shape[0] + 1
    if cluster_set_size >= number_of_observations:
        return 0
    elif cluster_set_size <= 1:
        return linkage[-1, 2] * 1.001
    else:
        return linkage[-cluster_set_size, 2] * 1.001


def plot_range_of_clusters(times, clusters, clusters_limits, ax=None):
    if ax is None:
        ax = plt.gca()

    if len(clusters.shape) <= 1:
        clusters = [clusters]
    for i in range(len(clusters)):
        number_of_clusters = len(set(clusters[i]))
        (cmap, number_of_colors) = (plt.cm.tab20, 20) if number_of_clusters > 10 else (plt.cm.tab10, 10)
        y = clusters_limits[i] * np.ones(len(times))
        ax.scatter(times, y, c=clusters[i], cmap=cmap, vmin=1, vmax=number_of_colors)

    ax.set_ylim([-1, ax.get_ylim()[1]])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    sb.despine(ax=ax)


def plot_time_clusters_right_axis(cluster_set_sizes, labels, ax):
    if ax is None:
        ax = plt.gca()

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(cluster_set_sizes)

    ax_right.set_yticklabels(labels)
    sb.despine(ax=ax_right, right=False)
