import numpy as np
import pandas as pd
import networkx


def read_static_network(filepath, format, separator=None):
    # Static network can be an adjacancy matrix saved as a numpy array or 'static_network_separator' separated
    # file, or in any format supported by networkx's read/write functionality, with the exception of JSON data
    # formats. See https://networkx.github.io/documentation/stable/reference/readwrite/index.html for more details.

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


def edge_list_from_temporal_edge_data(networkx_graph, filepath, separator):
    # Temporal edge data should be a 'temporal_data_separator' separated file with three or four columns,
    # representing source node, target node, time and (optionally) weight, respectively. If the weight column is
    # omitted then all weights are taken to be 1. For nodes i and j, if the static network contains an edge ij,
    # and at time t the edge ij has weight w at least 'threshold', then the temporal network will contain an
    # edge ij at time t of weight w.

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


def edge_list_from_temporal_node_data(networkx_graph, filepath, separator, combine_node_weights):
    # Temporal node data should be a 'temporal_data_separator' separated file with two or three columns,
    # representing node, time and (optionally) weight, respectively. If the weight column is omitted then all
    # weights are taken to be 1. For nodes i and j, if the static network contains an edge ij, and at time t the
    # the result r of passing the weight of i and the weight of j to 'combine_node_weights' is at least
    # 'threshold', then the temporal network will contain an edge ij at time t of weight r.
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
