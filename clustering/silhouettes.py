import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.cluster.hierarchy as sch
from sklearn import metrics


def compute_silhouettes_for_max_cluster_range(clusters, distance_matrix, max_cluster_range, number_of_time_points):
    max_cluster_range_length = len(max_cluster_range)

    # One set of flat clusters per number in max_cluster_range
    flat_clusters = np.zeros((max_cluster_range_length, number_of_time_points))
    numbers_of_clusters = np.zeros(max_cluster_range_length)

    # One set of silhouette samples per number in max_cluster_range
    silhouette_samples = np.zeros((max_cluster_range_length, number_of_time_points))
    average_silhouettes = np.zeros(max_cluster_range_length)

    for i, max_clusters in enumerate(max_cluster_range):
        flat_clusters[i] = sch.fcluster(clusters, max_clusters, criterion='maxclust')
        numbers_of_clusters[i] = len(set(flat_clusters[i]))

        if numbers_of_clusters[i] > 1:
            average_silhouettes[i] = metrics.silhouette_score(distance_matrix, flat_clusters[i], metric="precomputed")
            silhouette_samples[i] = metrics.silhouette_samples(distance_matrix, flat_clusters[i], metric="precomputed")

    return flat_clusters, numbers_of_clusters, silhouette_samples, average_silhouettes


def plot_average_silhouettes(average_silhouettes, numbers_of_clusters, max_clusters_range, labels, ylim, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(average_silhouettes, numbers_of_clusters, 'ko-')

    ax.set_ylabel("Actual # clusters")
    ax.set_xlabel("Average silhouette")
    ax.set_xlim(xmax=1.1)
    ax.set_ylim(ylim)
    ax.set_yticks(max_clusters_range)
    ax.set_yticklabels(labels)


def plot_silhouette_sample(silhouette_sample, clusters, silhouette_avg, ax=None):
    if ax is None:
        ax = plt.gca()

    n_clust = len(set(clusters))
    if n_clust > 10:
        sb.set_palette("tab20")
    else:
        sb.set_palette("tab10")

    y_lower = 1
    for i in range(1, n_clust + 1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_sample[clusters == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #         color = plt.cm.nipy_spectral(float(i) / n_clust)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=f"C{i - 1}", edgecolor=f"C{i - 1}", alpha=1)

        # Label the silhouette plots with their cluster numbers at the middle
        #         ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        hpad = 1
        y_lower = y_upper + hpad  # 10 for the 0 samples

    ax.axvline(x=silhouette_avg, c='k', ls='--')

    # ax.set_ylim(1, len(clusters) + n_clust*hpad)

    # ax.set_title(f"The silhouette plot for the {n_clust} clusters.")
    #     ax.set_xlabel("The silhouette coefficient values")
    #     ax.set_ylabel("Cluster label")
    #     ax.set_yticks([])
    sb.despine()
