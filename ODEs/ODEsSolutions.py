import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sb
from labellines import labelLines

from ODEs.ODEs import normed
from ODEs.xppcall import xpprun

comparators = {'minima': np.less, 'maxima': np.greater}


class ODEsSolutions:
    def __init__(self, filepath, start_time=None, end_time=None):
        # If given, start time should be inclusive and end_time should be exclusive.
        times_and_solutions, variables = xpprun(filepath, clean_after=True)

        # Since our temporal network has had times shifted to start at zero, do the same here.
        self.true_times = times_and_solutions[start_time:end_time, 0]
        self.times = self.true_times if not start_time else self.true_times - start_time
        self.solutions = times_and_solutions[start_time:end_time, 1:]
        self.variables = variables

    def series(self, variable):
        return self.solutions[:, self.variables.index(variable)]

    def relative_optima(self, variable, optima_type):
        series = self.series(variable)
        optima_times = scipy.signal.argrelextrema(series, comparators[optima_type])
        optima = series[optima_times]
        return optima, optima_times

    def plot_relative_optima(self, variable, optima_type, ax=None):
        ax.plot(self.times, self.series(variable), 'o-')
        mass_minima, mass_minima_times = self.relative_optima(variable, optima_type)
        ax.plot(self.times[mass_minima_times], mass_minima, 'ro')

    def plot_concentrations(self, variables, ax=None, norm=False, labels_xvals=None):
        if ax is None:
            ax = plt.gca()

        for variable in variables:
            y = normed(self.series(variable)) if norm else self.series(variable)
            ax.plot(self.times, y, label=variable)

        if not labels_xvals:
            # Add evenly-spaced labels
            labels_interval = len(self.times) // (len(variables) + 1)
            labels_xvals = [self.times[labels_interval * (i + 1)] for i in range(len(variables))]
        labelLines(ax.get_lines(), zorder=2.5, xvals=labels_xvals)
