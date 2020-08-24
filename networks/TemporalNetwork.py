import pathlib
import pickle
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
    def from_snapshots_numpy_array(_class, snapshots, node_ids_to_names=None):
        # 'snapshots' should be a numpy.array with dimensions (time, nodes, nodes)
        array = np.swapaxes(snapshots, 0, 2)
        return TemporalNetwork.from_numpy_array(array, node_ids_to_names)

    @classmethod
    def from_numpy_array(_class, array, node_ids_to_names=None):
        # 'array' should be a numpy.array with dimensions (nodes, nodes, time)
        teneto_network = teneto.TemporalNetwork(from_array=array)
        return _class(teneto_network, node_ids_to_names=node_ids_to_names)

    @classmethod
    def from_edge_list_dataframe(_class, edges, normalise=False, threshold=0, binary=False):
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

        if normalise:
            edges['weight'] = (edges['weight'] / edges['weight'].max())
        edges = edges[edges['weight'] >= threshold]
        if binary:
            edges[edges['weight'] > 0]['weight'] = 1

        # Replace node names with numeric values
        nodes = sorted(set(edges[['i', 'j']].values.flatten()))
        names_to_ids = {key: i for i, key in enumerate(nodes)}
        ids_to_names = {i: key for i, key in enumerate(nodes)}

        # ToDo - this is now the slowest step in creating a TemporalNetwork; speed it up?
        edges = edges.replace(names_to_ids)

        edges = edges.sort_values('t')
        edges.reset_index(drop=True, inplace=True)
        start_time = edges['t'][0]
        if start_time != 0:
            # For compatibility with teneto, shift all times so that we start at time 0
            edges['t'] -= start_time

        teneto_network = teneto.TemporalNetwork(from_df=edges)
        return _class(teneto_network, start_time, ids_to_names)

    @classmethod
    def from_static_network_and_edge_list_dataframe(
            _class,
            static_network,
            edges,
            normalise=False,
            threshold=0,
            binary=False):

        # 'static_network' should be a networkx.Graph.

        # 'edges' should be a pandas.DataFrame with with three or four columns, representing source node, target
        # node, time and (optionally) weight, respectively. If the weight column is omitted then all weights are
        # taken to be 1.

        # If 'normalise' is True, all weights will be divided through by the max weight across all edges.

        # For nodes i and j, if the static network contains an edge ij, and at time t the edge ij has weight w at
        # least 'threshold' (after normalisation), then the temporal network will contain an edge ij at time t of
        # weight w.

        # If 'binary' is True, then any positive edges that exceed the threshold will be set to 1.

        if len(edges.columns) == 3:
            edges['w'] = 1
        elif len(edges.columns) != 4:
            raise ValueError('Edge list should have either 3 or 4 columns.')
        edges.columns = ['i', 'j', 't', 'w']

        static_network_edges = pd.DataFrame(static_network.edges)
        static_network_edges.columns = ['static_i', 'static_j']
        edges = edges.merge(static_network_edges, how='left', left_on=['i', 'j'], right_on=['static_i', 'static_j'])
        edges = edges.merge(static_network_edges, how='left', left_on=['i', 'j'], right_on=['static_j', 'static_i'])

        missing_edges_indices = edges['static_i_x'].isnull() & edges['static_i_y'].isnull()
        if missing_edges_indices.any():
            missing_edges = edges[missing_edges_indices]
            missing_edges = list(zip(missing_edges['i'], missing_edges['j']))
            print(f'WARNING: The following edges were not found in the static network:\n{missing_edges}')

        # Only keep edges that are present in the static network
        edges = edges[~missing_edges_indices][['i', 'j', 't', 'w']]
        return _class.from_edge_list_dataframe(edges, normalise, threshold, binary)

    @classmethod
    def from_static_network_and_node_list_dataframe(
            _class,
            static_network,
            nodes,
            combine_node_weights=lambda x, y: x * y,
            normalise=False,
            threshold=0,
            binary=False):

        # 'static_network' should be a networkx.Graph.

        # 'nodes' should be a pandas.DataFrame with two or three columns, representing node, time and (optionally)
        # weight, respectively. If the weight column is omitted then all weights are taken to be 1.

        # If 'normalise' is True, all weights will be divided through by the max weight across all edges.

        # For nodes i and j, if the static network contains an edge ij, and at time t the edge ij has weight r at
        # least 'threshold' (after normalisation), then the temporal network will contain an edge ij at time t of
        # weight r. Here r is the result of passing the weight of i and the weight of j to 'combine_node_weights'

        # Note that 'combine_node_weights' is applied to whole columns at a time, for efficiency. Therefore any
        # unvectorizable lambda functions will throw an error.

        # If 'binary' is True, then any positive edges that exceed the threshold will be set to 1.

        if len(nodes.columns) == 2:
            nodes['w'] = 1
        elif len(nodes.columns) != 3:
            raise ValueError('Node list must have either 2 or 3 columns.')
        nodes.columns = ['i', 't', 'w']

        static_network_nodes = pd.DataFrame(static_network.nodes)
        static_network_nodes.columns = ['static_i']
        nodes = nodes.merge(static_network_nodes, how='left', left_on='i', right_on='static_i')

        missing_nodes_indices = nodes['static_i'].isnull()
        if missing_nodes_indices.any():
            missing_nodes = nodes[missing_nodes_indices]
            print(f'WARNING: The following nodes were not found in the static network:\n{missing_nodes["i"].values}')

        # Only keep nodes that are present in the static network
        nodes = nodes[~missing_nodes_indices][['i', 't', 'w']]
        # Merge to form temporal edge data
        edges = nodes.merge(nodes, left_on='t', right_on='t', suffixes=('_1', '_2'))
        # Remove duplicates and self-loops
        edges = edges[edges['i_1'] < edges['i_2']]
        # Add column for combined weight
        edges['r'] = combine_node_weights(edges['w_1'], edges['w_2'])
        edges = edges[['i_1', 'i_2', 't', 'r']]

        return _class.from_edge_list_dataframe(edges, normalise, threshold, binary)


    @classmethod
    def from_static_network_and_node_table_dataframe(
            _class,
            static_network,
            node_table,
            combine_node_weights=lambda x, y: x * y,
            normalise=False,
            threshold=0,
            binary=False):

        # 'static_network' should be a networkx.Graph.

        # 'temporal_node_table' should be a pandas.DataFrame whose columns represent the nodes of the graph,
        # and whose rows contain the temporal data for those nodes. The DataFrame should be indexed by time points,
        # which should be numeric values. For other parameters, see 'from_static_network_and_node_list_dataframe'.

        def get_node_list(node_table, node):
            node_list = node_table[node].to_frame()
            node_list['i'] = node
            node_list['t'] = node_list.index
            node_list = node_list[['i', 't', node]]
            node_list.columns = ['i', 't', 'w']
            return node_list

        node_lists = [get_node_list(node_table, node) for node in node_table.columns]
        node_list = pd.concat(node_lists, ignore_index=True)

        return _class.from_static_network_and_node_list_dataframe(
            static_network, node_list, combine_node_weights, normalise, threshold, binary)
