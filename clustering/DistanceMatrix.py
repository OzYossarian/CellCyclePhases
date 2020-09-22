import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


class DistanceMatrix:
    def __init__(self, full, condensed, metric):
        """
        Parameters
        __________
        full - the distance matrix whose (i,j)th entry holds the distance between items i and j of the dataset whose
            distances we're interested in.
        condensed - a vector created by flattening the upper triangular half of the full matrix above. Its usefulness
            stems from the fact that a distance matrix created from an 'undirected' dataset will be symmetric.
        metric - the distance metric used to create this matrix
        """

        self.full = full
        self.condensed = condensed
        self.metric = metric

    def plot_heatmap(self, ax=None, triangular=True, cmap="YlGnBu"):
        """Plot this distance matrix as a heatmap

        Parameters
        __________
        ax - the matplotlib axes on which to plot
        triangular - whether or not to omit the upper triangular entries; only useful if matrix is symmetric
        cmap - the desired colour map to use for the plot
        """

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
