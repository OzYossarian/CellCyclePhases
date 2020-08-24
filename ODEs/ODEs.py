import numpy as np
import matplotlib.pyplot as plt


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


def normed(x):
    return x / np.max(x)
