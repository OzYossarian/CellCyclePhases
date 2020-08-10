import pathlib
import pickle
import networkx
import numpy as np
import pandas as pd
import teneto


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
    def from_static_network_file(
            _class,
            static_network_filepath,
            static_network_format,
            static_network_separator=None,
            temporal_edge_data_filepath=None,
            temporal_node_data_filepath=None,
            temporal_data_separator=None,
            threshold=1,
            combine_node_weights=lambda x, y: x * y):

        # Static network can be an adjacancy matrix saved as a numpy array or 'static_network_separator' separated
        # file, or in any format supported by networkx's read/write functionality, with the exception of JSON data
        # formats. See https://networkx.github.io/documentation/stable/reference/readwrite/index.html for more details,
        # and see the function read_static_network for options for the parameter static_network_format.

        # Only one of temporal edge data or temporal node data should be provided.

        # Temporal edge data should be a 'temporal_data_separator' separated file with three or four columns,
        # representing source node, target node, time and (optionally) weight, respectively. If the weight column is
        # omitted then all weights are taken to be 1. For nodes i and j, if the static network contains an edge ij,
        # and at time t the edge ij has weight w at least 'threshold', then the temporal network will contain an
        # edge ij at time t of weight w.

        # Temporal node data should be a 'temporal_data_separator' separated file with two or three columns,
        # representing node, time and (optionally) weight, respectively. If the weight column is omitted then all
        # weights are taken to be 1. For nodes i and j, if the static network contains an edge ij, and at time t the
        # the result r of passing the weight of i and the weight of j to 'combine_node_weights' is at least
        # 'threshold', then the temporal network will contain an edge ij at time t of weight r.

        if not temporal_edge_data_filepath and not temporal_node_data_filepath:
            raise ValueError('No temporal data provided')
        elif temporal_edge_data_filepath and temporal_node_data_filepath:
            raise ValueError('Only one of temporal edge data or temporal node data should be provided')

        networkx_graph = read_static_network(
            static_network_filepath,
            static_network_format,
            separator=static_network_separator)

        if temporal_node_data_filepath:
            edges = read_temporal_node_data(
                networkx_graph, temporal_data_separator, temporal_node_data_filepath, combine_node_weights)
        else:
            edges = read_temporal_edge_data(networkx_graph, temporal_data_separator, temporal_edge_data_filepath)

        # Only keep edges meeting threshold criteria
        edges = edges[edges['w'] >= threshold]
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


def read_temporal_edge_data(networkx_graph, separator, filepath):
    edges = pd.read_csv(filepath, sep=separator, engine='python')
    if len(edges.columns) == 3:
        edges['w'] = 1
    edges.columns = ['i', 'j', 't', 'w']

    missing_edges_indices = edges.apply(lambda edge: not networkx_graph.has_edge(edge['i'], edge['j']), axis=1)
    missing_edges = edges[missing_edges_indices]
    if not missing_edges.empty:
        missing_edges = list(zip(missing_edges["i"], missing_edges["j"]))
        print(f'WARNING: The following edges were not found in the static network:\n{missing_edges}')

    # Only keep edges that are present in the static network
    edges = edges[~missing_edges_indices]
    return edges


def read_temporal_node_data(networkx_graph, separator, filepath, combine_node_weights):
    nodes = pd.read_csv(filepath, sep=separator, engine='python')
    if len(nodes.columns) == 2:
        nodes['w'] = 1
    nodes.columns = ['i', 't', 'w']

    missing_nodes_indices = nodes.apply(lambda node: not networkx_graph.has_node(node['i']), axis=1)
    missing_nodes = nodes[missing_nodes_indices]
    if not missing_nodes.empty:
        print(f'WARNING: The following nodes were not found in the static network:\n{missing_nodes["i"].values}')

    # Only keep nodes that are present in the static network
    nodes = nodes[~missing_nodes_indices]
    # Merge to form temporal edge data
    edges = nodes.merge(nodes, left_on='t', right_on='t', suffixes=('_1', '_2'))
    # Remove duplicates and self-loops
    edges = edges[edges['i_1'] < edges['i_2']]
    # Add column for combined weight
    edges['r'] = edges.apply(lambda edge: combine_node_weights(edge['w_1'], edge['w_2']), axis=1)
    edges = edges[['i_1', 'i_2', 't', 'r']]
    edges.columns = ['i', 'j', 't', 'w']
    return edges


def read_static_network(filepath, format, **kwargs):
    separator = kwargs['separator'] if 'separator' in kwargs else None
    if format == 'adjacency_matrix_numpy':
        return networkx.from_numpy_array(np.load(filepath, allow_pickle=True))
    elif format == 'adjacency_matrix':
        return networkx.from_pandas_adjacency(pd.read_csv(filepath, sep=separator, engine='python'))
    elif format == 'adjacency_list':
        return networkx.read_adjlist(filepath, delimiter=separator)
    elif format == 'adjecency_list_multiline':
        return networkx.read_multiline_adjlist(filepath, delimiter=separator)
    elif format == 'edge_list':
        return networkx.read_edgelist(filepath, delimiter=separator)
    elif format == 'GEXF':
        return networkx.read_gexf(filepath)
    elif format == 'GML':
        return networkx.read_gml(filepath)
    elif format == 'pickle':
        return networkx.read_gpickle(filepath)
    elif format == 'GraphML':
        return networkx.read_graphml(filepath)
    elif format == 'LEDA':
        return networkx.read_leda(filepath)
    elif format == 'YAML':
        return networkx.read_yaml(filepath)
    elif format == 'graph6':
        return networkx.read_graph6(filepath)
    elif format == 'sparse6':
        return networkx.read_sparse6(filepath)
    elif format == 'Pajek':
        return networkx.read_pajek(filepath)
    elif format == 'shapefile':
        return networkx.read_shp(filepath)
    else:
        raise ValueError(f'Unrecognised static network format {format}')
