import pandas as pd
import teneto


# ToDo: provide multiple constructors - e.g. create from file, from pandas.DataFrame, from numpy.array, etc
class TemporalNetwork:
    def __init__(self, edges):
        # Columns must be named i, j and t, with optional weight column
        number_of_columns = edges.shape[1]
        if number_of_columns == 3:
            columns = ['i', 'j', 't']
        elif number_of_columns == 4:
            columns = ['i', 'j', 't', 'weight']
        else:
            raise ValueError('List of edges requires either 3 or 4 columns')
        edges.columns = columns

        # Replace strings with numeric values
        nodes = sorted(set(edges[['i', 'j']].values.flatten()))
        node_ids = {key: i for i, key in enumerate(nodes)}
        edges = edges.replace(node_ids)

        self._time_points = sorted(set(edges['t']))
        start_time = self._time_points[0]
        if start_time != 0:
            # For compatibility with teneto, shift all times so that we start at time 0.
            edges['t'] -= start_time
            self._time_points_starting_at_zero = sorted(set(edges['t']))

        self.network = teneto.TemporalNetwork(from_df=edges, nodelabels=nodes)

        # Expose relevant properties/methods of underlying teneto network - add more as needed
        self.T = self.network.T
        self.df_to_array = self.network.df_to_array

    @classmethod
    def from_file(_class, filepath, separator):
        edges = pd.read_csv(filepath, sep=separator, engine='python')
        return _class(edges)

    def time_points(self, starting_at_zero):
        if starting_at_zero and hasattr(self, '_time_points_starting_at_zero'):
            return self._time_points_starting_at_zero
        else:
            return self._time_points
