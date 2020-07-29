import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import drawing
from drawing.utils import display_name


def plot_silhouettes_samples(cluster_sets, columns):
    total_subplots = len(cluster_sets)
    rows = (total_subplots // columns) + (0 if total_subplots % columns == 0 else 1)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(10, 2 * rows))

    for i, cluster_set in enumerate(cluster_sets):
        ax = axs.flatten()[i]
        title = f'{display_name(cluster_set.limit_type)} = {cluster_set.limit}'
        subtitle = f'({int(cluster_set.size)} clusters)'
        plot_silhouette_samples(cluster_set, ax=ax, title=f'{title}\n{subtitle}')

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
        if cluster_set.silhouette.samples.size > 0:
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
