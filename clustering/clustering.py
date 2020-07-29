import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


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