import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.cluster.hierarchy as sch
import seaborn as sb

from clustering import clustering


def plot_dendrogram_from_temporal_network(
        temporal_network,
        distance_type,
        method,
        ax=None,
        max_distance=None,
        leaf_rotation=90,
        leaf_font_size=8,
        title=""):

    clusters = clustering.linkage(temporal_network, distance_type, method)
    plot_dendrogram_from_clusters(clusters, ax, max_distance, leaf_rotation, leaf_font_size, title)


def plot_dendrogram_from_clusters(
        clusters,
        ax=None,
        max_distance=None,
        leaf_rotation=90,
        leaf_font_size=8,
        title=""):

    if ax is None:
        ax = plt.gca()

    sch.dendrogram(
        clusters,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=max_distance,
        above_threshold_color='black',
        ax=ax)

    if max_distance is not None:
        # Add dotted horizontal line at max distance
        ax.axhline(y=max_distance, c='grey', ls='--', zorder=1)

    ax.set_title(title, weight="bold")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Time")


def plot_scatter_of_phases_from_temporal_network(
        temporal_network,
        distance_type,
        method,
        max_clusters,
        number_of_colours,
        ax=None,
        title=""):

    clusters = clustering.linkage(temporal_network, distance_type, method)
    flat_clusters = sch.fcluster(clusters, max_clusters, criterion='maxclust')
    times = temporal_network.time_points(starting_at_zero=True)
    plot_scatter_of_phases_from_flat_clusters(flat_clusters, times, ax, number_of_colours, title)


def plot_scatter_of_phases_from_flat_clusters(flat_clusters, times, number_of_colours, ax=None, title=""):
    ax.scatter(times, times * 0, c=flat_clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
    ax.set_yticks([])
    sb.despine(ax=ax, left=True)
    ax.grid(axis='x')
    ax.set_title(title)


def plot_time_clusters(times, clusters, ax=None, cmap=plt.cm.tab10):
    if ax == None:
        ax = plt.gca()

    n_colors = 10
    n_plots = len(clusters.shape)
    n_t = len(times)

    if n_plots > 1:
        for i, clusters_i in enumerate(clusters):
            n_clust = len(set(clusters_i))

            if n_clust > 10:
                cmap = plt.cm.tab20
                n_colors = 20
            else:
                cmap = plt.cm.tab10
                n_colors = 10

            ax.scatter(times, (i + 1) * np.ones(n_t), c=clusters_i,
                       cmap=cmap, vmin=1, vmax=n_colors)
    else:
        n_clust = len(set(clusters))
        ax.scatter(times, 0 * np.ones(n_t), c=clusters,
                   cmap=cmap, vmin=1, vmax=n_colors)


def plot_events(ax=None):
    if ax == None:
        ax = plt.gca()

    y_pos = 1.01 * ax.get_ylim()[1]

    events_chen = [33, 84, 36, 100]
    event_chen_names = ['bud', 'spn', 'ori', 'mass']
    for i, event in enumerate(events_chen):
        ax.axvline(x=event, c='k', label=event_chen_names[i], zorder=-1)
        ax.text(event, y_pos, event_chen_names[i],  # transform=ax.transAxes,
                fontsize='small', rotation=90, va='bottom', ha='center')

    events = ['START', 'E3']
    events_times = [5, 70]
    for i, event in enumerate(events):
        ax.axvline(x=events_times[i], c='k', ls='--', label=event[i], zorder=-1)
        ax.text(events_times[i], y_pos, events[i],  # transform=ax.transAxes,
                fontsize='small', rotation=90, va='bottom', ha='center')


def plot_phases(ax=None, y_pos=None):
    if ax == None:
        ax = plt.gca()
    if y_pos == None:
        y_pos = 1.01 * ax.get_ylim()[1]

    phases = np.array([0, 35, 70, 78, 100])
    phases_mid = (phases[:-1] + phases[1:]) / 2
    phases_labels = ['G1', 'S', 'G2', 'M']

    for i in range(len(phases) - 1):
        ax.axvspan(xmin=phases[i], xmax=phases[i + 1], ymin=0, ymax=0.1, color='k', alpha=+ 0.15 * i)

        ax.text(phases_mid[i], -1, phases_labels[i], fontweight='bold',
                va='bottom', ha='center')


def configure_colour_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
