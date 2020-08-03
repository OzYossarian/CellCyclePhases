import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


class DistanceMatrix:
    def __init__(self, full, condensed, metric):
        self.full = full
        self.condensed = condensed
        self.metric = metric

    def plot_heatmap(self, ax=None, triangular=True, cmap="YlGnBu"):
        if ax is None:
            ax = plt.gca()

        if triangular:
            mask = np.zeros_like(self.full)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None
        with sb.axes_style("white"):
            sb.heatmap(self.full, ax=ax, mask=mask, square=True, cmap=cmap)
