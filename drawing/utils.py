import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.cluster.hierarchy as sch


def label_subplot_grid_with_shared_axes(rows, columns, total_subplots, xlabel, ylabel, fig, axs):
    if rows > 1:
        axs_left = axs[:, 0]
    else:
        axs_left = [axs[0]]

    for ax in axs_left:
        ax.set_ylabel(ylabel)

    for ax in axs.flatten()[-columns:]:
        ax.set_xlabel(xlabel)

    # size_of_extra_row = total_subplots % columns
    # if size_of_extra_row != 0 and rows > 1:
    #     blank_axs = axs[-1, (size_of_extra_row + 1):]
    #     above_blank_axs = axs[-2, (size_of_extra_row + 1):]
    #     for labels in above_blank_axs.get_xaxis().get_majorticklabels():
    #         labels.set_visible(True)
    #     fig.delaxes(blank_axs)


def configure_colour_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
