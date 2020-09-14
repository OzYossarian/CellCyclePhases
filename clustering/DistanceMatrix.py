import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


class DistanceMatrix:
    def __init__(self, full, condensed, metric):
        self.full = full
        # A distance matrix created from an undirected graph will be symmetric, so it's useful to have a condensed
        # version, creating by flattening the upper triangular half of the matrix into a vector
        self.condensed = condensed
        # Record the distance metric used to create this matrix
        self.metric = metric

    def plot_heatmap(self, ax=None, triangular=True, cmap="YlGnBu"):
        if ax is None:
            ax = plt.gca()

        # A distance matrix created from an undirected graph will be symmetric, so a triangular heatmap omits
        # redundant data in this case
        if triangular:
            mask = np.zeros_like(self.full)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None
        with sb.axes_style("white"):
            sb.heatmap(self.full, ax=ax, mask=mask, square=True, cmap=cmap)
