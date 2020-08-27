import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


# Think twice before accessing directly; the method display_name can be used instead.
_display_names = {
    'maxclust': 'Max # clusters',
    'distance': 'Distance threshold'
}


def label_subplot_grid_with_shared_axes(rows, columns, total_subplots, xlabel, ylabel, fig, axes):
    if rows > 1:
        axes_left = axes[:, 0]
    else:
        axes_left = [axes[0]]
    for ax in axes_left:
        ax.set_ylabel(ylabel)

    size_of_extra_row = total_subplots % columns

    if size_of_extra_row != 0 and rows > 1:
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
    return _display_names[key] if key in _display_names else key


def configure_sch_color_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])


def adjust_margin(ax=None, top=0, bottom=0, left=0, right=0):
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
