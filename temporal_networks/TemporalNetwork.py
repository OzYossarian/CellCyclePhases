import numpy as np
import pandas as pd
import teneto


# ToDo: provide multiple constructors - e.g. create from file, from pandas.DataFrame, from numpy.array, etc
class TemporalNetwork:
    def __init__(self, teneto_network, time_shift):
        self.teneto_network = teneto_network

        # Keep track of the 'true' times, even though we've shifted to start at 0
        self.times = np.array(sorted(set(self.teneto_network.network['t'])))
        self.true_times = self.times + time_shift

        # Expose relevant properties/methods of underlying teneto network - add more as needed
        self.T = self.teneto_network.T
        self.df_to_array = self.teneto_network.df_to_array

    @classmethod
    def from_dataframe(_class, edges):
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

        edges = edges.sort_values('t')
        start_time = edges['t'][0]
        if start_time != 0:
            # For compatibility with teneto, shift all times so that we start at time 0
            edges['t'] -= start_time

        return _class(teneto.TemporalNetwork(from_df=edges), time_shift=start_time)

    @classmethod
    def from_file(_class, filepath, separator):
        edges = pd.read_csv(filepath, sep=separator, engine='python')
        return TemporalNetwork.from_dataframe(edges)
