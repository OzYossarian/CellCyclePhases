import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def plot_dendrogram_from_clusters(
        clusters,
        number_of_clusters,
        ax=None,
        leaf_rotation=90,
        leaf_font_size=6,
        title=""):

    if ax is None:
        ax = plt.gca()

    distance_threshold = get_distance_threshold(clusters, number_of_clusters)
    sch.dendrogram(
        clusters,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=distance_threshold,
        above_threshold_color='black',
        ax=ax)

    ax.axhline(y=distance_threshold, c='grey', ls='--', zorder=1)
    ax.set_title(title, weight="bold")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Time points")


def plot_scatter_of_phases_from_flat_clusters(flat_clusters, times, number_of_colours, ax=None, title=""):
    ax.scatter(times, times * 0, c=flat_clusters, cmap=cm.tab10, vmin=1, vmax=number_of_colours)
    ax.set_yticks([])
    sb.despine(ax=ax, left=True)
    ax.grid(axis='x')
    ax.set_title(title, weight="bold")


def get_distance_threshold(clusters, number_of_clusters):
    number_of_observations = clusters.shape[0] + 1
    if number_of_clusters >= number_of_observations:
        return 0
    elif number_of_clusters <= 1:
        return clusters[-1, 2] * 1.001
    else:
        return clusters[-number_of_clusters, 2] * 1.001


def plot_time_clusters(times, flat_clusters, ax=None):
    if ax is None:
        ax = plt.gca()

    number_of_plots = len(flat_clusters.shape)
    if number_of_plots > 1:
        for i in range(len(flat_clusters)):
            number_of_clusters = len(set(flat_clusters[i]))
            (cmap, number_of_colors) = (plt.cm.tab20, 20) if number_of_clusters > 10 else (plt.cm.tab10, 10)
            y = (i + 1) * np.ones(len(times))
            ax.scatter(times, y, c=flat_clusters[i], cmap=cmap, vmin=1, vmax=number_of_colors)
    else:
        ax.scatter(times, 0 * np.ones(len(times)), c=flat_clusters, cmap=plt.cm.tab10, vmin=1, vmax=10)

    ax.set_ylabel("Max # clusters")
    ax.set_xlabel("Times (min)")

    ax.set_xticks(range(0, 100 + 5, 10))
    ax.set_ylim([-1, ax.get_ylim()[1]])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    sb.despine(ax=ax)


def plot_time_clusters_right_axis(max_cluster_range, labels, ax):
    if ax is None:
        ax = plt.gca()

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(max_cluster_range)

    ax_right.set_yticklabels(labels)
    sb.despine(ax=ax_right, right=False)
