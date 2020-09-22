import numpy as np
import scipy
import matplotlib.pyplot as plt
from labellines import labelLines

from temporal.xppcall import xpprun

comparators = {'minima': np.less, 'maxima': np.greater}


class TemporalData:
    """Class representing any sort of temporal data for a specified list of variables

    Since teneto's TemporalNetwork class requires all times to start at zero, we again have the concept of 'true' times
    as well as times offset to start at zero.
    """

    def __init__(self, temporal_data, variables, times, true_times):
        """
        Parameters
        __________
        temporal_data - a numpy array whose columns are the variables we're interested in and whose rows are the
            temporal data for these variables
        variables - a python list (not a numpy array) of variable names
        times - a numpy array based on true_times but with values shifted to start at zero
        true_times - a numpy array of time points
        """

        self.temporal_data = temporal_data
        self.variables = variables
        self.times = times
        self.true_times = true_times

    @classmethod
    def from_ODEs(class_, filepath, start_time=None, end_time=None, xpp_alias='xppaut'):
        """Create a TemporalData object by solving a system of ODEs

        Uses the solver XPP. XPP must be installed and executable on your PATH under the name xpp_alias

        Parameters
        __________
        filepath - the file specifying the system of ODEs to solve
        start_time - start of time period from which to take temporal data; set to None to start at beginning. If
            given, should be inclusive.
        end_time - end of time period from which to take temporal data; set to None to end at the end. If given,
            should be exclusive.
        xpp_alias - the name of the XPP executable on your PATH.
        """

        times_and_series, variables = xpprun(filepath, xppname=xpp_alias, clean_after=True)

        # Since our temporal network has had times shifted to start at zero, do the same here.
        # If given, start time should be inclusive and end_time should be exclusive.
        true_times = times_and_series[start_time:end_time, 0]
        times = true_times if not start_time else true_times - start_time
        temporal_data = times_and_series[start_time:end_time, 1:]
        return class_(temporal_data, variables, times, true_times)

    def series(self, variable):
        """Get temporal data for a particular variable"""
        return self.temporal_data[:, self.variables.index(variable)]

    def relative_optima(self, variable, optima_type):
        """Get relative optima for a particular variable

        Parameters
        __________
        variable - the name of the variable whose optima we want
        optima_type - 'minima' or 'maxima'

        Returns
        _______
        optima - the value of the variable at its optima
        optima_times - the times at which these optima occur
        """

        series = self.series(variable)
        optima_times = scipy.signal.argrelextrema(series, comparators[optima_type])
        optima = series[optima_times]
        return optima, optima_times

    def plot_relative_optima(self, variable, optima_type, ax=None, use_true_times=True):
        """Plot relative optima for a prticular variable

        Parameters
        __________
        variable - the name of the variable whose optima we want to plot
        optima_type - 'minima' or 'maxima'
        ax - the matplotlib axes on which to plot
        use_true_times - whether to use the 'actual' times or offset the times so that they start at zero
        """

        if ax is None:
            ax = plt.gca()

        times = self.true_times if use_true_times else self.times
        ax.plot(times, self.series(variable), 'o-')
        mass_minima, mass_minima_times = self.relative_optima(variable, optima_type)
        ax.plot(times[mass_minima_times], mass_minima, 'ro')

    def plot_series(self, variables, ax=None, norm=False, add_labels=True, labels_xvals=None, use_true_times=True):
        """Plot particular variables' values over time

        Parameters
        __________
        variables - iterable; the names of the variable whose values we want to plot
        ax - the matplotlib axes on which to plot
        norm - boolean; whether or not to normalise the time series by dividing through by the max
        add_labels - boolean; whether to label the variables when plotting
        labels_xvals - the positions along the x-axis at which to place the variables' labels (if using). If set to
            None, labels will be placed at regular intervals along x-axis.
        use_true_times - whether to use the 'actual' times or offset the times so that they start at zero
        """

        if ax is None:
            ax = plt.gca()
        times = self.true_times if use_true_times else self.times

        for variable in variables:
            y = normed(self.series(variable)) if norm else self.series(variable)
            ax.plot(times, y, label=variable)

        if add_labels:
            if not labels_xvals:
                # Add evenly-spaced labels
                labels_interval = len(times) // (len(variables) + 1)
                labels_xvals = [times[labels_interval * (i + 1)] for i in range(len(variables))]
            labelLines(ax.get_lines(), zorder=2.5, xvals=labels_xvals)


def normed(x):
    return x / np.max(x)