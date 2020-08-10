import pathlib
import pickle
import numpy as np
import pandas as pd
import teneto

from networks import static_networks


class TemporalNetwork:
    def __init__(self, teneto_network, time_shift=0, node_ids_to_names=None):
        self.teneto_network = teneto_network
        self.time_shift = time_shift
        self.node_ids_to_names = node_ids_to_names

        # Keep track of the 'true' times, even though we've shifted to start at 0
        if teneto_network.sparse:
            self.times = np.array(sorted(set(self.teneto_network.network['t'])))
        else:
            self.times = np.array(range(teneto_network.T))
        self.true_times = self.times + time_shift

        # Expose relevant methods of underlying teneto network - add more as needed
        self.T = self.teneto_network.T

    def get_snapshots(self):
        array = self.teneto_network.df_to_array() if self.sparse() else self.teneto_network.network
        snapshots = np.swapaxes(array, 0, 2)
        return snapshots

    def sparse(self):
        return self.teneto_network.sparse

    def node_name(self, id):
        if self.node_ids_to_names is None:
            return id
        else:
            return self.node_ids_to_names[id]

    @classmethod
    def from_snapshots(_class, snapshots, node_ids_to_names=None):
        # 'snapshots' should be a numpy.array with dimensions (time, nodes, nodes)
        array = np.swapaxes(snapshots, 0, 2)
        return TemporalNetwork.from_array(array, node_ids_to_names)

    @classmethod
    def from_array(_class, array, node_ids_to_names=None):
        # 'array' should be a numpy.array with dimensions (nodes, nodes, time)
        teneto_network = teneto.TemporalNetwork(from_array=array)
        return _class(teneto_network, node_ids_to_names=node_ids_to_names)

    @classmethod
    def from_edge_list_dataframe(_class, edges):
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
    def from_edge_list_file(_class, filepath, separator=None):
        edges = pd.read_csv(filepath, sep=separator, engine='python')
        return TemporalNetwork.from_edge_list_dataframe(edges)

    @classmethod
    def from_snapshots_file(_class, snapshots_filepath, node_ids_to_names_filepath=None):
        snapshots = load_file(snapshots_filepath)
        node_ids_to_names = load_file(node_ids_to_names_filepath)
        if isinstance(node_ids_to_names, np.ndarray):
            # Extract the dictionary from the numpy array
            node_ids_to_names = node_ids_to_names.item()
        return _class.from_snapshots(snapshots, node_ids_to_names)

    @classmethod
    def from_static_network(
            _class,
            static_network,
            temporal_edge_data_filepath=None,
            temporal_node_data_filepath=None,
            temporal_data_separator=None,
            threshold=0,
            combine_node_weights=lambda x, y: x * y,
            binary=False,
            normalise=False):

        # 'static_network' must be an instance of networkx.Graph.
        # Temporal data file should be as described in the function edge_list_from_temporal_node_data or
        # edge_list_from_temporal_edge_data.
        # If 'normalise' is True, all weights will be divided through by the max weight.
        # Only edges with weight at least 'threshold' (AFTER normalising but BEFORE binarying) will be kept.
        # If 'binary' is True, all edges with positive weight will be reassigned weight 1.

        if temporal_node_data_filepath:
            edges = static_networks.edge_list_from_temporal_node_data(
                static_network, temporal_node_data_filepath, temporal_data_separator, combine_node_weights)
        elif temporal_edge_data_filepath:
            edges = static_networks.edge_list_from_temporal_edge_data(
                static_network, temporal_edge_data_filepath, temporal_data_separator)
        else:
            raise ValueError('Provide exactly one of temporal edge data or temporal node data')

        if normalise:
            max_weight = edges['w'].max()
            edges['w'] = (edges['w'] / max_weight)
        edges = edges[edges['w'] >= threshold]
        if binary:
            edges['w'] = 1

        edges.reset_index(drop=True, inplace=True)
        return _class.from_edge_list_dataframe(edges)


def load_file(filepath):
    file_type = pathlib.Path(filepath).suffix.lower()
    if file_type == '.npy':
        loaded = np.load(filepath, allow_pickle=True)
    elif file_type in ['.pkl', '.pickle']:
        loaded = pickle.load(filepath)
    else:
        raise ValueError(f'Unknown file type "{file_type}" - consider loading the file yourself then '
                         f'using a different constructor')
    return loaded
