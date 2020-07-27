import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import drawing
from clustering import clustering


def plot_average_silhouettes_and_clusters(silhouettes, max_cluster_range, times, title):
    gridspec_kw = {"width_ratios": [9, 2]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw=gridspec_kw)

    numbers_of_clusters_differences = np.diff(silhouettes.numbers_of_clusters)
    labels = [
        int(number_of_clusters) if (i == 0 or numbers_of_clusters_differences[i - 1] != 0) else ''
        for i, number_of_clusters
        in enumerate(silhouettes.numbers_of_clusters)]

    clustering.plot_time_clusters(times, silhouettes.clusters, ax=ax1)
    clustering.plot_time_clusters_right_axis(silhouettes.numbers_of_clusters, labels, ax=ax1)
    plot_average_silhouettes(silhouettes, max_cluster_range, labels, ylim=ax1.get_ylim(), ax=ax2)

    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.4, top=0.8)

    return fig, (ax1, ax2)


def plot_average_silhouettes(silhouettes, max_clusters_range, labels, ylim, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(silhouettes.averages, silhouettes.numbers_of_clusters, 'ko-')

    ax.set_ylabel("Actual # clusters")
    ax.set_xlabel("Average silhouette")
    ax.set_xlim(xmax=1.1)
    ax.set_ylim(ylim)
    ax.set_yticks(silhouettes.numbers_of_clusters)
    ax.set_yticklabels(labels)


def plot_silhouette_samples(silhouettes, columns):
    unique_numbers_of_clusters, indices_of_unique_numbers_of_clusters = \
        np.unique(silhouettes.numbers_of_clusters, return_index=True)

    # Omit the 1-cluster
    one_cluster_index = np.where(unique_numbers_of_clusters == 1)
    unique_numbers_of_clusters = np.delete(unique_numbers_of_clusters, one_cluster_index)
    indices_of_unique_numbers_of_clusters = np.delete(indices_of_unique_numbers_of_clusters, one_cluster_index)
    total_subplots = len(unique_numbers_of_clusters)

    rows = (total_subplots // columns) + (0 if total_subplots % columns == 0 else 1)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(10, 2 * rows))

    for i, unique_j in enumerate(indices_of_unique_numbers_of_clusters):
        ax = axs.flatten()[i]
        silhouette = silhouettes[unique_j]
        title = f"{int(silhouette.number_of_clusters)} clusters"
        plot_silhouette_sample(silhouette, ax=ax, title=title)

    xlabel, ylabel = 'Silhouette score', 'Ordered time points'
    drawing.utils.label_subplot_grid_with_shared_axes(rows, columns, total_subplots, xlabel, ylabel, fig, axs)

    return fig, axs


def plot_silhouette_sample(silhouette, ax=None, title=''):
    if ax is None:
        ax = plt.gca()

    if silhouette.number_of_clusters > 10:
        sb.set_palette("tab20")
    else:
        sb.set_palette("tab10")

    y_lower = 1
    for i, cluster in enumerate(np.unique(silhouette.clusters)):
        # Aggregate the silhouette scores for samples belonging to each cluster, and sort them
        # ToDo: calculate values upon creation of Silhouette(s)? Then store them as silhouette(s).values?
        silhouette_values = silhouette.sample[silhouette.clusters == cluster]
        silhouette_values.sort()

        cluster_size = silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        y = np.arange(y_lower, y_upper)
        ax.fill_betweenx(y, 0, silhouette_values, facecolor=f"C{i}", edgecolor=f"C{i}", alpha=1)

        # Compute the new y_lower for next plot
        vertical_padding = 1
        y_lower = y_upper + vertical_padding

    ax.set_title(title)
    ax.axvline(x=silhouette.average, c='k', ls='--')
    sb.despine()
