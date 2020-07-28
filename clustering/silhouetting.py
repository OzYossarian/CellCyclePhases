import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import drawing
from clustering import clustering


def plot_average_silhouettes_and_clusters(cluster_sets, times, title):
    # ToDo - three subplots instead of two (or two plots on the second axes):
    # 1. Clusters over range of limits
    # 2. Average silhouette over range of limits
    # 3. Actual number of clusters (i.e. cluster set size) over range of limits

    gridspec_kw = {"width_ratios": [9, 2]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw=gridspec_kw)

    cluster_set_sizes_differences = np.diff(cluster_sets.sizes)
    labels = [
        int(size) if (i == 0 or cluster_set_sizes_differences[i - 1] != 0) else ''
        for i, size
        in enumerate(cluster_sets.sizes)]

    clustering.plot_range_of_clusters(times, cluster_sets.clusters, cluster_sets.limits, ax=ax1)
    clustering.plot_time_clusters_right_axis(cluster_sets.sizes, labels, ax=ax1)
    plot_average_silhouettes(cluster_sets, labels, ylim=ax1.get_ylim(), ax=ax2)

    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.4, top=0.8)

    return fig, (ax1, ax2)


def plot_average_silhouettes(cluster_sets, labels, ylim, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(cluster_sets.silhouettes.averages, cluster_sets.sizes, 'ko-')

    ax.set_ylabel("Actual # clusters")
    ax.set_xlabel("Average silhouette")
    ax.set_xlim(xmax=1.1)
    ax.set_ylim(ylim)
    ax.set_yticks(cluster_sets.sizes)
    ax.set_yticklabels(labels)


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
