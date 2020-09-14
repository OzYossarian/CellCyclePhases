import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from matplotlib.colors import ListedColormap


def label_subplot_grid_with_shared_axes(rows, columns, total_subplots, xlabel, ylabel, fig, axes):
    """
    Method to tidy up cases where we have a grid of plots with shared axes, e.g. by deleting unused subplots (if
    number of of subplots is not rectangular) and adding axes ticks.
    """
    if rows > 1:
        axes_left = axes[:, 0]
    else:
        axes_left = [axes[0]]
    for ax in axes_left:
        ax.set_ylabel(ylabel)

    # Bottom row will potentially have fewer subplots than all other rows.
    size_of_extra_row = total_subplots % columns

    if size_of_extra_row != 0 and rows > 1:
        # Delete blank subplots and add x-axis ticks to subplots on penultimate row above blank subplots
        blank_axes = axes[-1, size_of_extra_row:]
        above_blank_axes = axes[-2, size_of_extra_row:]
        axes_on_extra_row = axes[-1, :size_of_extra_row]
        for ax in blank_axes:
            fig.delaxes(ax)
        for ax in above_blank_axes:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(xlabel)
        for ax in axes_on_extra_row:
            ax.set_xlabel(xlabel)

    else:
        for ax in axes.flatten()[-columns:]:
            ax.set_xlabel(xlabel)


def display_name(key):
    names = {
        'maxclust': 'Max # clusters',
        'distance': 'Distance threshold'
    }
    return names[key] if key in names else key


def configure_sch_color_map(cmap):
    rgbs = cmap(np.linspace(0, 1, 10))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in rgbs])


def adjust_margin(ax=None, top=0, bottom=0, left=0, right=0):
    # Extend the margin of a plot by a percentage of its original width/height
    if ax is None:
        ax = plt.gca()

    if top or bottom:
        y_limits = ax.get_ylim()
        difference = y_limits[-1] - y_limits[0]
        new_y_limits = [y_limits[0] - difference * bottom, y_limits[-1] + difference * top]
        ax.set_ylim(new_y_limits)

    if left or right:
        x_limits = ax.get_xlim()
        difference = x_limits[-1] - x_limits[0]
        new_x_limits = [x_limits[0] - difference * left, x_limits[-1] + difference * right]
        ax.set_xlim(new_x_limits)


def get_extrema_of_binary_series(mask, times) :
    """return xmins and xmaxs for intervals in times"""
    binary = 1*mask
    slope = np.diff(binary)
    signs = slope[(slope != 0)]

    if np.all(binary==1):
        mins = [times[0]]
        maxs = [times[-1]]
    elif np.all(binary==0):
        mins = [times[0]]
        maxs = [times[0]]
    else:
        mins = list(times[1:][slope==1])
        maxs = list(times[:-1][slope==-1])

        if signs[0]==-1:
            mins = [times[0]] + mins

        if signs[-1]==1:
            maxs = maxs + [times[-1]]

    return mins, maxs