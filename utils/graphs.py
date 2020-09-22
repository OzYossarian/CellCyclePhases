import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def create_graph_from_interactions(filename, sheet, source, target):
    """Create a networkx.Graph from an excel sheet describing edges

    Parameters
    __________
    filename - path to the excel file
    sheet - name of the sheet within the excel file
    source - name of the column containing the source nodes
    target - name of the column containing the target nodes
    """

    interactions = pd.read_excel(filename, sheet_name=sheet)
    graph = nx.from_pandas_edgelist(interactions, source, target)
    return graph


def draw_graph(graph, ax=None, label_nodes=True, color='mediumseagreen'):
    """Basic graph drawing function

    Parameters
    __________
    graph - a networkx.Graph object
    ax - the matplotlib axes on which to draw the graph
    label_nodes - whether to label the nodes or just leave them as small circles
    color - color to use for the graph nodes and edges
    """

    if ax is None:
        ax = plt.gca()

    layout = graphviz_layout(graph, prog='fdp')
    _draw_graph(graph, layout, ax, label_nodes, color)


def _draw_graph(graph, layout, ax, label_nodes, color):
    # PRIVATE function for the parts of graph drawing that are common to multiple methods
    params = standard_node_params(color)
    nx.draw_networkx_nodes(graph, ax=ax, pos=layout, **params)
    nx.draw_networkx_edges(graph, ax=ax, pos=layout, **params)
    if label_nodes:
        nx.draw_networkx_labels(graph, ax=ax, pos=layout, **params)


def highlight_subgraphs(graphs, colors, ax=None, label_nodes=True):
    """Draw multiple nested subgraphs on the same axes

    Parameters
    __________
    graphs - an iterable of networkx.Graph objects
    colors - an iterable of names of colors, one for each of the graphs in 'graphs'
    ax - the matplotlib axes to plot on
    label_nodes - whether or not to label the graph nodes or leave them as circles
    """

    if ax is None:
        ax = plt.gca()

    layout = graphviz_layout(graphs[0], prog='fdp')
    for graph, color in zip(graphs, colors):
        _draw_graph(graph, layout, ax, label_nodes, color)


def graph_size_info(graph):
    return f"{len(graph)} nodes and {len(graph.edges)} edges"


def standard_node_params(color):
    return {
        'node_color': color,
        'edge_color': color,
        'font_color': 'k',
        'edgecolors': 'k',
        'node_size': 150,
        'bbox': dict(facecolor=color, edgecolor='black', boxstyle='round, pad=0.2', alpha=1)
    }
