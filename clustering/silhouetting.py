import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import drawing


def plot_average_silhouettes(silhouettes, max_clusters_range, labels, ylim, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(silhouettes.averages, silhouettes.numbers_of_clusters, 'ko-')

    ax.set_ylabel("Actual # clusters")
    ax.set_xlabel("Average silhouette")
    ax.set_xlim(xmax=1.1)
    ax.set_ylim(ylim)
    ax.set_yticks(max_clusters_range)
    ax.set_yticklabels(labels)


def plot_silhouette_samples(silhouettes, columns):
    unique_numbers_of_clusters, indices_of_unique_numbers_of_clusters = \
        np.unique(silhouettes.numbers_of_clusters, return_index=True)

    # Omit the 1-cluster
    total_subplots = len(unique_numbers_of_clusters) - 1

    rows = (total_subplots // columns) + (0 if total_subplots % columns == 0 else 1)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(10, 2 * rows))

    for i, unique_j in enumerate(indices_of_unique_numbers_of_clusters):
        ax = axs.flatten()[i - 1]
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
    for i in range(1, int(silhouette.number_of_clusters) + 1):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        # ToDo: calculate values upon creation of Silhouette(s)? Then store them as silhouette(s).values?
        silhouette_values = silhouette.sample[silhouette.clusters == i]
        silhouette_values.sort()

        cluster_size = silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        y = np.arange(y_lower, y_upper)
        ax.fill_betweenx(y, 0, silhouette_values, facecolor=f"C{i - 1}", edgecolor=f"C{i - 1}", alpha=1)

        # Compute the new y_lower for next plot
        # ToDo - figure out what hpad does and give it a more descriptive name.
        hpad = 1
        y_lower = y_upper + hpad  # 10 for the 0 samples

    ax.set_title(title)
    ax.axvline(x=silhouette.average, c='k', ls='--')
    sb.despine()
