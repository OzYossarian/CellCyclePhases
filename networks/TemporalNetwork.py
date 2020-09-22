import numpy as np
import pandas as pd
import teneto


class TemporalNetwork:
    """A wrapper class around teneto's TemporalNetwork class

    The main reason for doing this is that teneto requires that the time points in the temporal network start at zero.
    Since it is often more useful to use the actual time points from the underlying dataset, we create a wrapper class,
    exposing/adding a few other methods/properties too.
    """

    def __init__(self, teneto_network, time_shift=0, node_ids_to_names=None):
        """
        Parameters
        __________
        teneto_network - a teneto.TemporalNetwork object
        time_shift - how much we've shifted the data's time points by in order to start at zero
        node_ids_to_names - a dictionary whose keys are node IDs (integers) and values (of any type) are more
            descriptive names for nodes.
        """

        self.teneto_network = teneto_network
        self.time_shift = time_shift
        self.node_ids_to_names = node_ids_to_names

        # Keep track of the 'true' times, even though we've shifted to start at 0
        if teneto_network.sparse:
            self.times = np.array(sorted(set(self.teneto_network.network['t'])))
        else:
            self.times = np.array(range(teneto_network.T))
        self.true_times = self.times + time_shift


    def T(self):
        """Returns: the number of time points in the temporal network"""

        return self.teneto_network.T

    def get_snapshots(self):
        """Returns: the representation of the network as a sequence of adjacency matrices, one for each time point"""

        array = self.teneto_network.df_to_array() if self.sparse() else self.teneto_network.network
        # Make 'time' the 0th axis
        snapshots = np.swapaxes(array, 0, 2)
        return snapshots

    def sparse(self):
        return self.teneto_network.sparse

    def node_name(self, id):
        """Get descriptive node name

        Parameters
        __________
        id - the integer ID of the node

        Returns
        _______
        The more descriptive name for this node, defaulting to 'id' if no descriptive name exists.
        """

        if self.node_ids_to_names is None or id not in self.node_ids_to_names:
            return id
        else:
            return self.node_ids_to_names[id]

    @classmethod
    def from_snapshots_numpy_array(_class, snapshots, node_ids_to_names=None):
        """Create a TemporalNetwork from snapshots

        Parameters
        __________
        snapshots -  a numpy.array with dimensions (time, nodes, nodes)
        node_ids_to_names - a dictionary whose keys (integers) are node IDs and values (of any type) are more
            descriptive names for nodes.
        """

        array = np.swapaxes(snapshots, 0, 2)
        return _class.from_numpy_array(array, node_ids_to_names)

    @classmethod
    def from_numpy_array(_class, array, node_ids_to_names=None):
        """Create a TemporalNetwork from an array representation of the network

        Parameters
        __________
        array - a numpy.array with dimensions (nodes, nodes, time)
        node_ids_to_names - a dictionary whose keys (integers) are node IDs and values (of any type) are more
            descriptive names for nodes.
        """

        teneto_network = teneto.TemporalNetwork(from_array=array)
        return _class(teneto_network, node_ids_to_names=node_ids_to_names)

    @classmethod
    def from_edge_list_dataframe(_class, edges, normalise=None, threshold=0, binary=False):
        """Create a TemporalNetwork from a DataFrame of edges over time

        Parameters
        __________
        edges - a pandas.DataFrame with columns representing source node, target node, time and (optionally) weight
        normalise - a value determining what (if any) normalisation is applied. If 'normalise' is 'global', all weights
            will be divided through by the max weight across all edges. If 'normalise' is 'local', all weights
            corresponding to an edge (i,j) at some time will be divided through by the max weight of the edge (i,j)
            across all times. To skip normalisation, set to None.
        threshold - any edges with weight < 'threshold' (after normalising) will not be included in the temporal network
        binary - if True, all positive weights (after thresholding) will be set to 1. If False, does nothing.
        """

        number_of_columns = edges.shape[1]
        if number_of_columns == 3:
            edges['weight'] = 1
        elif number_of_columns != 4:
            raise ValueError('Edge list requires either 3 or 4 columns')
        edges.columns = ['i', 'j', 't', 'weight']

        if normalise == 'global':
            min_weight = edges['weight'].min()
            difference = edges['weight'].max() - min_weight
            edges['weight'] = (edges['weight'] - min_weight) / difference
        if normalise == 'local':
            # Sort first two columns; we only care about *unordered* pairs (i,j), not ordered pairs.
            edges[['i', 'j']] = np.sort(edges[['i', 'j']], axis=1)
            grouped = edges.groupby(['i', 'j'])['weight']
            maxes = grouped.transform('max')
            mins = grouped.transform('min')
            edges['weight'] = (edges['weight'] - mins) / (maxes - mins)
            # In cases where max = min we'll have a division by zero error.
            edges['weight'] = edges['weight'].fillna(0.5)
        edges = edges[edges['weight'] >= threshold]
        if binary:
            edges.loc[edges['weight'] > 0, 'weight'] = 1

        edges, ids_to_names = replace_nodes_with_ids(edges)
        edges = edges.sort_values('t')
        start_time = edges['t'].iloc[0]
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
            normalise=None,
            threshold=0,
            binary=False,
            static_edges_default=None):
        """Create a TemporalNetwork from a static network and a DataFrame of temporal edges

        Given a static network and a set of edges across different times, we can create a temporal network by including
        temporal edge (i,j,t) if edge the static network contains an edge (i,j).

        Parameters
        __________
        static_network - a networkx.Graph object representing the underlying static network
        edges - a pandas.DataFrame with columns representing source node, target node, time and (optionally) weight
        normalise - a value determining what (if any) normalisation is applied. If 'normalise' is 'global', all weights
            will be divided through by the max weight across all edges. If 'normalise' is 'local', all weights
            corresponding to an edge (i,j) at some time will be divided through by the max weight of the edge (i,j)
            across all times. To skip normalisation, set to None.
        threshold - any edges with weight < 'threshold' (after normalising) will not be included in the temporal network
        binary - if True, all positive weights (after thresholding) will be set to 1. If False, does nothing.
        static_edges_default - if there are edges in the static network that aren't present in 'edges' then
            'static_edges_default' determines what to do with these. If set to None, these static edges are simply
            ignored. If set to a numerical value k, these static edges are given weight k across all time points.
        """

        if len(edges.columns) == 3:
            edges['w'] = 1
        elif len(edges.columns) != 4:
            raise ValueError('Edge list should have either 3 or 4 columns.')
        edges.columns = ['i', 'j', 't', 'w']

        # Our fastest option throughout this method is to merge DataFrames. So begin by converting static network's
        # edges to a DataFrame.
        static_network_edges = pd.DataFrame(static_network.edges)
        static_network_edges.columns = ['static_i', 'static_j']

        # Sort the first two columns of each DataFrame so that edges are pairs (i, j) with i <= j. This allows
        # us to merge DataFrames together without regard for the direction of the edges (i.e. whether we have an
        # edge (i,j) or (j,i)).
        static_network_edges[['static_i', 'static_j']] = np.sort(static_network_edges[['static_i', 'static_j']], axis=1)
        edges[['i', 'j']] = np.sort(edges[['i', 'j']], axis=1)

        # Merge edges from static network into our temporal edge list.
        if static_edges_default is None:
            # We only want edges that appear in both the temporal edge list and the static network. We could do this
            # with an inner join, but we want to inform the user of edges in the temporal edge list that weren't
            # matched to any in the static network. So we use a left join initially.
            edges = edges.merge(static_network_edges, how='left', left_on=['i', 'j'], right_on=['static_i', 'static_j'])
        else:
            # We want edges that appear in both the temporal edge list and the static network, and we additionally
            # want to add constant temporal data for edges in the static network but not in the temporal edge list.
            edges = edges.merge(static_network_edges, how='outer', left_on=['i', 'j'], right_on=['static_i', 'static_j'])
            # Rows in the DataFrame with no value in column i are those from the static network that had no
            # corresponding edge in the temporal data.
            message = f'The following static edges have no temporal data and will have default weight ' \
                      f'{static_edges_default} across all time points'
            edges, static_missing_edges = remove_missing_edges(edges, 'i', ['static_i', 'static_j'], message)

        message = 'The following edges were not found in the static network'
        edges, _ = remove_missing_edges(edges, 'static_i', ['i', 'j'], message)
        edges = edges[['i', 'j', 't', 'w']]

        if static_edges_default is not None and not static_missing_edges.empty:
            # Add default weights across all timepoints for the static edges that have no temporal data.
            times = edges['t'].drop_duplicates().to_frame().assign(w=static_edges_default)
            temporal_static_edges = static_missing_edges[['static_i', 'static_j']].assign(w=static_edges_default)
            temporal_static_edges = temporal_static_edges.merge(times, on='w')[['static_i', 'static_j', 't', 'w']]
            temporal_static_edges.columns = ['i', 'j', 't', 'w']
            edges = pd.concat([edges[['i', 'j', 't', 'w']], temporal_static_edges], ignore_index=True)

        return _class.from_edge_list_dataframe(edges, normalise, threshold, binary)

    @classmethod
    def from_static_network_and_node_list_dataframe(
            _class,
            static_network,
            nodes,
            combine_node_weights=lambda x, y: x * y,
            normalise=None,
            threshold=0,
            binary=False,
            static_edges_default=None):
        """Create a TemporalNetwork from a static network and a DataFrame of temporal nodes

        Given a static network and a set of nodes across different times, we can create a temporal network by including
        temporal edge (i,j,t) if edge the static network contains an edge (i,j).

        Parameters
        __________
        static_network - a networkx.Graph object representing the underlying static network
        nodes - a pandas.DataFrame with columns representing node, time and (optionally) weight
        combine_node_weights - a lambda determining how to combine two nodes' weights together to give the weight of
            the corresponding edge. NOTE: this is applied to whole columns at a time, for efficiency. Therefore any
            unvectorizable lambda functions will raise an exception.
        normalise - a value determining what (if any) normalisation is applied. If 'normalise' is 'global', all weights
            will be divided through by the max weight across all edges. If 'normalise' is 'local', all weights
            corresponding to an edge (i,j) at some time will be divided through by the max weight of the edge (i,j)
            across all times. To skip normalisation, set to None.
        threshold - any edges with weight < 'threshold' (after normalising) will not be included in the temporal network
        binary - if True, all positive weights (after thresholding) will be set to 1. If False, does nothing.
        static_edges_default - if there are edges in the static network that aren't present in 'edges' then
            'static_edges_default' determines what to do with these. If set to None, these static edges are simply
            ignored. If set to a numerical value k, these static edges are given weight k across all time points.
        """

        if len(nodes.columns) == 2:
            nodes['w'] = 1
        elif len(nodes.columns) != 3:
            raise ValueError('Node list must have either 2 or 3 columns.')
        nodes.columns = ['i', 't', 'w']

        # Our fastest option throughout this method is to merge DataFrames. So begin by converting static network's
        # nodes to a DataFrame.
        static_network_nodes = pd.DataFrame(static_network.nodes)
        static_network_nodes.columns = ['static_i']
        nodes = nodes.merge(static_network_nodes, how='left', left_on='i', right_on='static_i')

        # Rows in the DataFrame with no value in static_i are exactly those that had no corresponding node in the
        # static network. So remove these edges, informing the user of this fact.
        missing_nodes_indices = nodes['static_i'].isnull()
        if missing_nodes_indices.any():
            missing_nodes = nodes[missing_nodes_indices]
            print(f'WARNING: The following nodes were not found in the static network:\n{missing_nodes["i"].values}')

        # Only keep nodes that are present in the static network
        nodes = nodes[~missing_nodes_indices][['i', 't', 'w']]
        # Merge nodes with themselves to form temporal edge data
        edges = nodes.merge(nodes, left_on='t', right_on='t', suffixes=('_1', '_2'))
        # Remove duplicates and self-loops
        edges = edges[edges['i_1'] < edges['i_2']]
        # Add column for combined weight
        edges['r'] = combine_node_weights(edges['w_1'], edges['w_2'])
        edges = edges[['i_1', 'i_2', 't', 'r']]

        return _class.from_static_network_and_edge_list_dataframe(
            static_network, edges, normalise, threshold, binary, static_edges_default)

    @classmethod
    def from_static_network_and_node_table_dataframe(
            _class,
            static_network,
            node_table,
            combine_node_weights=lambda x, y: x * y,
            normalise=None,
            threshold=0,
            binary=False,
            static_edges_default=None):
        """Create a TemporalNetwork from a static network and a DataFrame of temporal nodes

        Given a static network and a set of nodes across different times, we can create a temporal network by including
        temporal edge (i,j,t) if edge the static network contains an edge (i,j).

        Parameters
        __________
        static_network - a networkx.Graph object representing the underlying static network
        node_table - a pandas.DataFrame whose columns represent the nodes of the graph, and whose rows
            contain the temporal data for those nodes. The DataFrame should be indexed by time points, which should
            be numeric values.
        combine_node_weights - a lambda determining how to combine two nodes' weights together to give the weight of
            the corresponding edge. NOTE: this is applied to whole columns at a time, for efficiency. Therefore any
            unvectorizable lambda functions will raise an exception.
        normalise - a value determining what (if any) normalisation is applied. If 'normalise' is 'global', all weights
            will be divided through by the max weight across all edges. If 'normalise' is 'local', all weights
            corresponding to an edge (i,j) at some time will be divided through by the max weight of the edge (i,j)
            across all times. To skip normalisation, set to None.
        threshold - any edges with weight < 'threshold' (after normalising) will not be included in the temporal network
        binary - if True, all positive weights (after thresholding) will be set to 1. If False, does nothing.
        static_edges_default - if there are edges in the static network that aren't present in 'edges' then
            'static_edges_default' determines what to do with these. If set to None, these static edges are simply
            ignored. If set to a numerical value k, these static edges are given weight k across all time points.
        """

        def get_node_list(node_table, node):
            # Turn the column representing a particular node into a list of temporal edges.
            node_list = node_table[node].to_frame()
            node_list['i'] = node
            node_list['t'] = node_list.index
            node_list = node_list[['i', 't', node]]
            node_list.columns = ['i', 't', 'w']
            return node_list

        # For each node in the table, create a list of temporal edges for that node, then concatenate all such lists
        # together to create the total edge list.
        node_lists = [get_node_list(node_table, node) for node in node_table.columns]
        node_list = pd.concat(node_lists, ignore_index=True)

        return _class.from_static_network_and_node_list_dataframe(
            static_network, node_list, combine_node_weights, normalise, threshold, binary, static_edges_default)


def remove_missing_edges(edges, null_column, print_columns, message):
    """Removes unwanted edges from a DataFrame based on their value in a certain column

    Parameters
    __________
    edges - a DataFrame whose columns include null_column and every column in print_columns
    null_column - the name n of the column such that if a row has a null value in column n then it will be removed
    print_columns - the columns whose values to zip together when printing the list of edges removed
    message - a string to preface the list of edges removed

    Returns
    _______
    edges - the new dataframe with required edges removed
    missing_edges - a dataframe containing all the edges that were removed
    """

    missing_edges_indices = edges[null_column].isnull()
    missing_edges = edges[missing_edges_indices]
    if not missing_edges.empty:
        printable_missing = list(zip(*[missing_edges[column] for column in print_columns]))
        print(f'{message}:\n{printable_missing}')
        edges = edges[~missing_edges_indices]
    return edges, missing_edges


def replace_nodes_with_ids(edges):
    """Replace node names with integer IDs

    Parameters
    __________
    edges - a DataFrame with nodes in columns 'i' and 'j'

    Returns
    _______
    edges - the updated DataFrame with names replaced by IDs
    ids_to_names - a dictionary mapping the new IDs to the old names
    """

    unique_nodes = pd.concat([edges['i'], edges['j']]).to_frame().drop_duplicates(ignore_index=True)
    unique_nodes.reset_index(inplace=True)
    unique_nodes.columns = ['id', 'name']
    ids_to_names = unique_nodes.to_dict()['name']

    # Replace all occurrences of a node name with its new numeric ID. We do this using merges for efficiency purposes.
    # Merge unique nodes onto our original edge list twice - once for when the node is the source node of an edge, and
    # again for when the node is the target node of an edge.
    edges = edges.merge(unique_nodes, how='left', left_on='i', right_on='name')
    edges = edges.merge(unique_nodes, how='left', left_on='j', right_on='name')
    # Now restrict to four columns again, but now using node IDs instead of node names.
    edges = edges[['id_x', 'id_y', 't', 'weight']]
    edges.columns = ['i', 'j', 't', 'weight']

    return edges, ids_to_names
