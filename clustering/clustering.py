import numpy as np
import matplotlib as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch

from clustering.cluster_snapshots import compute_snapshot_distances


def plot_dendrogram(temporal_network, distance, method, ax, leaf_rotation=90, leaf_font_size=8):
    _, distance_matrix_condensed = compute_snapshot_distances(temporal_network, dist=distance)
    linked = sch.linkage(distance_matrix_condensed, method=method)
    sch.dendrogram(
        linked,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        above_threshold_color='black',
        ax=ax)
    if ax is not None:
        ax.set_ylabel("Distance")
        ax.set_xlabel("Temporal")


def configure_colour_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([plt.colors.rgb2hex(rgb[:3]) for rgb in cmap])