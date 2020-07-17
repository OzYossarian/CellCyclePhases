import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import scipy.cluster.hierarchy as shc
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sb

sb.set_context("paper")


def compute_snapshot_distances(tnet, dist='euclidean'):
    """
    Compute pairwise distance between snapshots

    parameters
    ----------
    - dist : string ('cosinesim', 'euclidean', 'euclidean_flat')

    returns
    -------

    """

    T = tnet.T
    dist_mat = np.zeros((T, T))

    snapshots = tnet.df_to_array()
    snapshots = np.swapaxes(snapshots, 0, 2)  # put time as zeroth axis
    snapshot_flat = snapshots.reshape(T, -1)  # each matrix is flattened, represented as a vector

    if dist == 'cosinesim':
        dist_mat = 1 - cosine_similarity(snapshot_flat, snapshot_flat)
        np.fill_diagonal(dist_mat, 0)  # otherwise, 1e-15 but negative values, cause problems later

    elif dist == 'euclidean' or dist == 'euclidean_flat':
        if dist == 'euclidean':
            pass
        elif dist == "euclidean_flat":
            snapshots = snapshot_flat

        for j in range(T):
            for i in range(j):  # fill upper triangle only
                dist_mat[i, j] = np.linalg.norm(snapshots[i] - snapshots[j])  # Eucledian distance

        dist_mat = dist_mat + dist_mat.T

    # extract condensed distance matrix needed for the clustering
    id_u = np.triu_indices(n=T, k=1)  # indices of upper triangle elements
    dist_mat_condensed = dist_mat[id_u]

    return dist_mat, dist_mat_condensed


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
