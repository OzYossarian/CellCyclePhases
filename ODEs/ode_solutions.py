

from ODEs.xppcall import xpprun


class OdeSolutions:
    def __init__(self, filepath, start_time, end_time):
        times_and_solutions, variables = xpprun(filepath, clean_after=True)
        self.times = times_and_solutions[start_time:end_time, 0]
        self.solutions = times_and_solutions[start_time:end_time, 1:]
        self.variables = variables
        self.start_time = start_time  # inclusive
        self.end_time = end_time  # exclusive
        # variables = [var.upper() for var in variables]
        # data = {var: series(var) for var in variables}

    def series(self, variable):
        return self.solutions[:, self.variables.index(variable)]
