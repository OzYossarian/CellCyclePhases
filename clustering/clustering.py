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


def get_distance_threshold(linkage, cluster_set_size):
    number_of_observations = linkage.shape[0] + 1
    if cluster_set_size >= number_of_observations:
        return 0
    elif cluster_set_size <= 1:
        return linkage[-1, 2] * 1.001
    else:
        return linkage[-cluster_set_size, 2] * 1.001


# ToDo - make plot_cluster_sets call this method
def plot_cluster_set(clusters, times, ax=None, title="", number_of_colours=10):
    ax.scatter(times, times * 0, c=clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
    ax.set_yticks([])
    sb.despine(ax=ax, left=True)
    ax.grid(axis='x')
    ax.set_title(title, weight="bold")


def plot_cluster_sets(times, cluster_sets, ax=None):
    if ax is None:
        ax = plt.gca()

    for cluster_set in cluster_sets:
        (cmap, number_of_colors) = (plt.cm.tab20, 20) if cluster_set.size > 10 else (plt.cm.tab10, 10)
        y = cluster_set.limit * np.ones(len(times))
        ax.scatter(times, y, c=cluster_set.clusters, cmap=cmap, vmin=1, vmax=number_of_colors)

    # Leave some space at the bottom in which to plot phases later
    limits_size = (cluster_sets.limits[-1] - cluster_sets.limits[0])
    ylim_bottom = cluster_sets.limits[0] - limits_size / 4
    ylim_top = cluster_sets.limits[-1] + limits_size / 12
    ax.set_ylim([ylim_bottom, ylim_top])

    ax.set_xlabel("Times (min)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    sb.despine(ax=ax)


def plot_cluster_sets_sizes(cluster_sets, ax):
    if ax is None:
        ax = plt.gca()

    ax.plot(cluster_sets.sizes, cluster_sets.limits, 'ko-')
    ax.set_xlabel("Actual # clusters")