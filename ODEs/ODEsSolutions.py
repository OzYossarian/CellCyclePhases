import numpy as np
import scipy

from ODEs.xppcall import xpprun

comparators = {'minima': np.less, 'maxima': np.greater}


class ODEsSolutions:
    def __init__(self, filepath, start_time, end_time):
        times_and_solutions, variables = xpprun(filepath, clean_after=True)
        self.times = times_and_solutions[start_time:end_time, 0]
        self.solutions = times_and_solutions[start_time:end_time, 1:]
        self.variables = variables
        self.start_time = start_time  # Inclusive
        self.end_time = end_time  # Exclusive

    def series(self, variable):
        return self.solutions[:, self.variables.index(variable)]

    def relative_optima(self, variable, optima_type):
        series = self.series(variable)
        optima_times = scipy.signal.argrelextrema(series, comparators[optima_type])
        optima = series[optima_times]
        return optima, optima_times
