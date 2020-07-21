import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.cluster.hierarchy as sch
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity


def compute_snapshot_distances(temporal_network, distance_type='euclidean'):
    """
    Compute pairwise distance between snapshots

    parameters
    ----------
    - dist : string ('cosinesim', 'euclidean', 'euclidean_flat')

    returns
    -------

    """

    T = temporal_network.T
    distance_matrix = np.zeros((T, T))

    snapshots = temporal_network.df_to_array()
    snapshots = np.swapaxes(snapshots, 0, 2)  # put time as zeroth axis
    snapshot_flat = snapshots.reshape(T, -1)  # each matrix is flattened, represented as a vector

    if distance_type == 'cosinesim':
        distance_matrix = 1 - cosine_similarity(snapshot_flat, snapshot_flat)
        np.fill_diagonal(distance_matrix, 0)  # otherwise, 1e-15 but negative values, cause problems later

    else:
        if distance_type == "euclidean_flat":
            snapshots = snapshot_flat
        for j in range(T):
            for i in range(j):  # fill upper triangle only
                distance_matrix[i, j] = np.linalg.norm(snapshots[i] - snapshots[j])  # Eucledian distance

        distance_matrix = distance_matrix + distance_matrix.T

    # Transform into condensed distance matrix needed for the clustering
    upper_triangular_indices = np.triu_indices(n=T, k=1)
    distance_matrix_condensed = distance_matrix[upper_triangular_indices]

    return distance_matrix, distance_matrix_condensed


def plot_dendrogram_from_temporal_network(
        temporal_network,
        distance_type,
        method,
        ax=None,
        max_distance=None,
        leaf_rotation=90,
        leaf_font_size=8,
        title=""):

    _, distance_matrix_condensed = compute_snapshot_distances(temporal_network, distance_type)
    clusters = sch.linkage(distance_matrix_condensed, method=method)
    plot_dendrogram_from_clusters(clusters, ax, max_distance, leaf_rotation, leaf_font_size, title)


def plot_dendrogram_from_clusters(
        clusters,
        ax=None,
        max_distance=None,
        leaf_rotation=90,
        leaf_font_size=6,
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
    ax.set_xlabel("Time points")


def plot_scatter_of_phases_from_temporal_network(
        temporal_network,
        distance_type,
        method,
        max_clusters,
        number_of_colours,
        ax=None,
        title=""):

    _, distance_matrix_condensed = compute_snapshot_distances(temporal_network, distance_type)
    clusters = sch.linkage(distance_matrix_condensed, method=method)
    flat_clusters = sch.fcluster(clusters, max_clusters, criterion='maxclust')

    times = temporal_network.time_points(starting_at_zero=True)
    plot_scatter_of_phases_from_flat_clusters(flat_clusters, times, ax, number_of_colours, title)


def plot_scatter_of_phases_from_flat_clusters(flat_clusters, times, number_of_colours, ax=None, title=""):
    ax.scatter(times, times * 0, c=flat_clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
    ax.set_yticks([])
    sb.despine(ax=ax, left=True)
    ax.grid(axis='x')
    ax.set_title(title, weight="bold")


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


def configure_colour_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])