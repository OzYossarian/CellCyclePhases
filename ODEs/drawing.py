import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def plot_concentrations(ode_solutions, variables, times, ax=None, norm=False):
    # ToDo - can we just use ode_solutions.times instead of times?
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


def plot_events(events, ax=None, y_pos=None):
    # An event is a triple (time, name, line_style), e.g.
    # events = [
    #     (5, 'START', '--'),
    #     (33, 'bud', None),
    #     (36, 'ori', None),
    #     (70, 'E3', '--'),
    #     (84, 'spn', None),
    #     (100, 'mass', None)
    # ]

    if ax is None:
        ax = plt.gca()
    if y_pos is None:
        y_pos = 1.01 * ax.get_ylim()[1]

    for event in events:
        time, name, line_style = event
        ax.axvline(x=time, c='k', ls=line_style, label=name, zorder=-1)
        ax.text(time, y_pos, name, fontsize='small', rotation=90, va='bottom', ha='center')


def plot_phases(phases, ax=None, y_pos=None):
    # A phase is a triple (start_time, end_time, name), e.g.
    # phases = [
    #     (0, 35, 'G1'),
    #     (35, 70, 'S'),
    #     (70, 78, 'G2'),
    #     (78, 100, 'M')
    # ]

    if ax is None:
        ax = plt.gca()

    for i, phase in enumerate(phases):
        start_time, end_time, name = phase
        # ToDo - better 'alpha' variable?
        ax.axvspan(xmin=start_time, xmax=end_time, ymin=0, ymax=0.1, color='k', alpha=+ 0.15 * i)
        ax.text((start_time + end_time)/2, -1, name, fontweight='bold', va='bottom', ha='center')


def normed(x):
    return x / np.max(x)
