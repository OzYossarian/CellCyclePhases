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
            nodes = pd.read_csv(temporal_node_data_filepath, sep=temporal_data_separator, engine='python')
            if len(nodes.columns) == 2:
                nodes['w'] = 1
            nodes.columns = ['i', 't', 'w']
            # Merge to form temporal edge data
            edges = nodes.merge(nodes, left_on='t', right_on='t', suffixes=('_1', '_2'))
            # Remove duplicates and self-loops
            edges = edges[edges['i_1'] < edges['i_2']]
            # Only keep rows meeting threshold criteria
            edges['r'] = edges.apply(lambda edge: combine_node_weights(edge['w_1'], edge['w_2']), axis=1)
            edges = edges[['i_1', 'i_2', 't', 'r']]
        else:
            edges = pd.read_csv(temporal_edge_data_filepath, sep=temporal_data_separator, engine='python')
            if len(edges.columns) == 3:
                edges['w'] = 1

        edges.columns = ['i', 'j', 't', 'w']
        edges = edges[edges['w'] >= threshold]
        edges = edges[edges.apply(lambda edge: networkx_graph.has_edge(edge['i'], edge['j']), axis=1)]
        edges.reset_index(drop=True, inplace=True)
        return _class.from_dataframe(edges)


def read_static_network(path, format, **kwargs):
    separator = kwargs['separator'] if 'separator' in kwargs else None
    if format == 'adjacency_matrix_numpy':
        return networkx.from_numpy_array(np.load(path, allow_pickle=True))
    elif format == 'adjacency_matrix':
        return networkx.from_pandas_adjacency(pd.read_csv(path, sep=separator, engine='python'))
    elif format == 'adjacency_list':
        return networkx.read_adjlist(path, delimiter=separator)
    elif format == 'adjecency_list_multiline':
        return networkx.read_multiline_adjlist(path, delimiter=separator)
    elif format == 'edge_list':
        return networkx.read_edgelist(path, delimiter=separator)
    elif format == 'GEXF':
        return networkx.read_gexf(path)
    elif format == 'GML':
        return networkx.read_gml(path)
    elif format == 'pickle':
        return networkx.read_gpickle(path)
    elif format == 'GraphML':
        return networkx.read_graphml(path)
    elif format == 'LEDA':
        return networkx.read_leda(path)
    elif format == 'YAML':
        return networkx.read_yaml(path)
    elif format == 'graph6':
        return networkx.read_graph6(path)
    elif format == 'sparse6':
        return networkx.read_sparse6(path)
    elif format == 'Pajek':
        return networkx.read_pajek(path)
    elif format == 'shapefile':
        return networkx.read_shp(path)
    else:
        raise ValueError(f'Unrecognised static network format {format}')
