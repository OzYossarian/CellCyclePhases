import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils import drawing


def plot_events(events, ax=None, y_pos=None, text_x_offset=0):
    if ax is None:
        ax = plt.gca()
    if y_pos is None:
        y_pos = 1.01 * ax.get_ylim()[1]
    if text_x_offset < 0:
        text_x_offset = -text_x_offset

    for event in events:
        time, name, line_style = event
        ax.axvline(x=time, c='k', ls=line_style, label=name, zorder=-1)
        text_x_pos = time - text_x_offset if time > 0 else time + text_x_offset
        ax.text(text_x_pos, y_pos, name, fontsize='small', rotation=90, va='bottom', ha='center')


def plot_phases(phases, ax=None, y_pos=None, ymin=0, ymax=1):
    if ax is None:
        ax = plt.gca()

    y_pos = y_pos if y_pos is not None else 1.01
    y_lim = ax.get_ylim()
    absolute_y_pos = y_lim[0] + y_pos * (y_lim[1] - y_lim[0])

    for i, phase in enumerate(phases):
        start_time, end_time, name = phase
        mid_time = (start_time + end_time)/2
        alpha_interval = 0.5 / len(phases)
        ax.axvspan(xmin=start_time, xmax=end_time, ymin=ymin, ymax=ymax, color='k', alpha=alpha_interval*(i+1))
        ax.text(mid_time, absolute_y_pos, name, fontweight='bold', va='center', ha='center')


def threshold_plot(x, y, threshold, color_below_threshold, color_above_threshold, ax=None):
    if ax is None:
        ax = plt.gca()

    # Create a colormap for red, green and blue and a norm to color
    # f' < -0.5 red, f' > 0.5 blue, and the rest green
    cmap = ListedColormap([color_below_threshold, color_above_threshold])
    norm = BoundaryNorm([np.min(y), threshold, np.max(y)], cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    line_collection = LineCollection(segments, cmap=cmap, norm=norm)
    line_collection.set_array(y)

    ax.add_collection(line_collection)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y)*1.1, np.max(y)*1.1)
    return line_collection


def plot_interval(mask, times, y=0, peak=None, color='k', ax=None, zorder=0):
    if ax is None:
        ax = plt.gca()

    xmins, xmaxs = drawing.get_extrema_of_binary_series(mask, times)
    rect_height = 0.5

    for xmin, xmax in zip(xmins, xmaxs):
        rect = patches.Rectangle((xmin, y), xmax-xmin, rect_height, fill=True, color=color, zorder=zorder)
        ax.add_patch(rect)
    if peak is not None:
        ax.plot(peak, y + rect_height / 2, 'r*')