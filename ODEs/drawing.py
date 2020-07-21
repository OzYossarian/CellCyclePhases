import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def plot_concentrations(ode_solutions, variables, times, ax=None, norm=False):
    if ax is None:
        ax = plt.gca()

    for variable in variables:
        if norm:
            ax.plot(times, normed(ode_solutions.series(variable)), label=variable)
        else:
            ax.plot(times, ode_solutions.series(variable), label=variable)

    ax.set_xlabel('Time (min)')
    if norm:
        ax.set_ylabel('Concentration (normed)')
    else:
        ax.set_ylabel('Concentration')

    sb.despine()

    return ax


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


def plot_phases(phases, ax=None, y_pos=None):
    if ax is None:
        ax = plt.gca()

    for i, phase in enumerate(phases):
        start_time, end_time, name = phase
        mid_time = (start_time + end_time)/2
        # ToDo - better 'alpha' variable?
        ax.axvspan(xmin=start_time, xmax=end_time, color='k', alpha=+ 0.15 * i)
        ax.text(mid_time, y_pos, name, fontweight='bold', va='bottom', ha='center')


def normed(x):
    return x / np.max(x)
