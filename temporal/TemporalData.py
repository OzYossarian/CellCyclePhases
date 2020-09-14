import numpy as np
import scipy
import matplotlib.pyplot as plt
from labellines import labelLines

from temporal.xppcall import xpprun

comparators = {'minima': np.less, 'maxima': np.greater}


class TemporalData:
    """ Class representing any sort of temporal data for a specified list of variables"""
    def __init__(self, temporal_data, variables, times, true_times):
        self.temporal_data = temporal_data
        # 'variables' should be a list (NOT, for example, a numpy array)
        self.variables = variables

        # teneto's TemporalNetwork class requires all times to start at 0. For compatibility, do the same here.
        self.times = times
        self.true_times = true_times

    @classmethod
    def from_ODEs(class_, filepath, start_time=None, end_time=None, xpp_alias='xppaut'):
        # Get temporal data by solving a system of ODEs, using the solver XPP. XPP must be installed and available on
        # your PATH - xpp_alias should be the name of XPP on this PATH.
        times_and_series, variables = xpprun(filepath, xppname=xpp_alias, clean_after=True)

        # Since our temporal network has had times shifted to start at zero, do the same here.
        # If given, start time should be inclusive and end_time should be exclusive.
        true_times = times_and_series[start_time:end_time, 0]
        times = true_times if not start_time else true_times - start_time
        temporal_data = times_and_series[start_time:end_time, 1:]
        return class_(temporal_data, variables, times, true_times)

    def series(self, variable):
        return self.temporal_data[:, self.variables.index(variable)]

    def relative_optima(self, variable, optima_type):
        series = self.series(variable)
        optima_times = scipy.signal.argrelextrema(series, comparators[optima_type])
        optima = series[optima_times]
        return optima, optima_times

    def plot_relative_optima(self, variable, optima_type, ax=None):
        ax.plot(self.times, self.series(variable), 'o-')
        mass_minima, mass_minima_times = self.relative_optima(variable, optima_type)
        ax.plot(self.times[mass_minima_times], mass_minima, 'ro')

    def plot_series(self, variables, ax=None, norm=False, add_labels=True, labels_xvals=None):
        if ax is None:
            ax = plt.gca()

        for variable in variables:
            y = normed(self.series(variable)) if norm else self.series(variable)
            ax.plot(self.times, y, label=variable)

        if add_labels:
            if not labels_xvals:
                # Add evenly-spaced labels
                labels_interval = len(self.times) // (len(variables) + 1)
                labels_xvals = [self.times[labels_interval * (i + 1)] for i in range(len(variables))]
            labelLines(ax.get_lines(), zorder=2.5, xvals=labels_xvals)


def normed(x):
    return x / np.max(x)