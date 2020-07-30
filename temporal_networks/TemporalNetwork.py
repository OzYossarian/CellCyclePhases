import numpy as np
import pandas as pd
import teneto


class TemporalNetwork:
    def __init__(self, teneto_network, time_shift=0, node_ids_to_names=None):
        self.teneto_network = teneto_network
        self.time_shift = time_shift
        self.node_ids_to_names = node_ids_to_names

        # Keep track of the 'true' times, even though we've shifted to start at 0
        self.times = np.array(sorted(set(self.teneto_network.network['t'])))
        self.true_times = self.times + time_shift

        # Expose relevant properties/methods of underlying teneto network - add more as needed
        self.T = self.teneto_network.T
        self.df_to_array = self.teneto_network.df_to_array

    def node_name(self, id):
        if self.node_ids_to_names is None:
            return id
        else:
            return self.node_ids_to_names[id]

    @classmethod
    def from_snapshots(_class, snapshots):
        # 'snapshots' should be a numpy.array with dimensions (time, nodes, nodes)
        array = np.swapaxes(snapshots, 0, 2)
        return TemporalNetwork.from_array(array)

    @classmethod
    def from_array(_class, array):
        # 'array' should be a numpy.array with dimensions (nodes, nodes, time)
        teneto_network = teneto.TemporalNetwork(from_array=array)
        return _class(teneto_network)

    @classmethod
    def from_dataframe(_class, edges):
        # 'edges' should be a pandas.DataFrame
        number_of_columns = edges.shape[1]
        # Columns must be named i, j and t, with optional weight column
        if number_of_columns == 3:
            columns = ['i', 'j', 't']
        elif number_of_columns == 4:
            columns = ['i', 'j', 't', 'weight']
        else:
            raise ValueError('List of edges requires either 3 or 4 columns')
        edges.columns = columns

        # Replace node names with numeric values
        nodes = sorted(set(edges[['i', 'j']].values.flatten()))
        names_to_ids = {key: i for i, key in enumerate(nodes)}
        ids_to_names = {i: key for i, key in enumerate(nodes)}
        edges = edges.replace(names_to_ids)

        edges = edges.sort_values('t')
        start_time = edges['t'][0]
        if start_time != 0:
            # For compatibility with teneto, shift all times so that we start at time 0
            edges['t'] -= start_time

        return _class(teneto.TemporalNetwork(from_df=edges), start_time, ids_to_names)

    @classmethod
    def from_file(_class, filepath, separator):
        edges = pd.read_csv(filepath, sep=separator, engine='python')
        return TemporalNetwork.from_dataframe(edges)
