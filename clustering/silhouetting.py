import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import drawing
from clustering import clustering
from drawing.utils import display_name


def plot_average_silhouettes_and_clusters(cluster_sets, cluster_limit_type, times, title):
    gridspec_kw = {"width_ratios": [3, 1, 2]}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3), gridspec_kw=gridspec_kw, sharey=True)

    clustering.plot_cluster_sets(times, cluster_sets, ax=ax1)
    plot_average_silhouettes(cluster_sets, ax=ax2)
    clustering.plot_cluster_sets_sizes(cluster_sets, ax=ax3)

    ax2.yaxis.set_tick_params(labelleft=True)
    ax3.yaxis.set_tick_params(labelleft=True)

    ax1.set_ylabel(display_name(cluster_limit_type))
    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.4, top=0.8)

    return fig, (ax1, ax2, ax3)


def plot_average_silhouettes(cluster_sets,ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(cluster_sets.silhouettes.averages, cluster_sets.limits, 'ko-')
    ax.set_xlabel("Average silhouette")
    ax.set_xlim((-0.1, 1.1))


def plot_silhouettes_samples(cluster_sets, columns):
    unique_numbers_of_clusters, indices_of_unique_numbers_of_clusters = \
        np.unique(cluster_sets.sizes, return_index=True)

    # Omit the 1-cluster
    one_cluster_index = np.where(unique_numbers_of_clusters == 1)
    unique_numbers_of_clusters = np.delete(unique_numbers_of_clusters, one_cluster_index)
    indices_of_unique_numbers_of_clusters = np.delete(indices_of_unique_numbers_of_clusters, one_cluster_index)
    total_subplots = len(unique_numbers_of_clusters)

    rows = (total_subplots // columns) + (0 if total_subplots % columns == 0 else 1)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(10, 2 * rows))

    for i, unique_j in enumerate(indices_of_unique_numbers_of_clusters):
        ax = axs.flatten()[i]
        cluster_set = cluster_sets[unique_j]
        title = f"{int(cluster_set.size)} clusters"
        plot_silhouette_samples(cluster_set, ax=ax, title=title)

    xlabel, ylabel = 'Silhouette score', 'Ordered time points'
    drawing.utils.label_subplot_grid_with_shared_axes(rows, columns, total_subplots, xlabel, ylabel, fig, axs)

    return fig, axs


def plot_silhouette_samples(cluster_set, ax=None, title=''):
    if ax is None:
        ax = plt.gca()

    if cluster_set.size > 10:
        sb.set_palette("tab20")
    else:
        sb.set_palette("tab10")

    y_lower = 1
    for i, cluster in enumerate(np.unique(cluster_set.clusters)):
        # Aggregate the silhouette scores for samples belonging to each cluster, and sort them
        # ToDo: calculate values upon creation of Silhouette(s)? Then store them as silhouette(s).values?
        silhouette_values = cluster_set.silhouette.samples[cluster_set.clusters == cluster]
        silhouette_values.sort()

        silhouette_size = silhouette_values.shape[0]
        y_upper = y_lower + silhouette_size
        y = np.arange(y_lower, y_upper)
        ax.fill_betweenx(y, 0, silhouette_values, facecolor=f"C{i}", edgecolor=f"C{i}", alpha=1)

        # Compute the new y_lower for next plot
        vertical_padding = 1
        y_lower = y_upper + vertical_padding

    ax.set_title(title)
    ax.axvline(x=cluster_set.silhouette.average, c='k', ls='--')
    sb.despine()
