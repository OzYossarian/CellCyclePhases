import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def create_graph_from_interactions(filename, sheet, source, target):
    interactions = pd.read_excel(filename, sheet_name=sheet)
    graph = nx.from_pandas_edgelist(interactions, source, target)
    return graph


def draw_graph(graph, ax=None, label_nodes=True, color='mediumseagreen'):
    if ax is None:
        ax = plt.gca()

    layout = graphviz_layout(graph, prog='fdp')
    _draw_graph(graph, layout, ax, label_nodes, color)


def _draw_graph(graph, layout, ax, label_nodes, color):
    params = standard_node_params(color)
    nx.draw_networkx_nodes(graph, ax=ax, pos=layout, **params)
    nx.draw_networkx_edges(graph, ax=ax, pos=layout, **params)
    if label_nodes:
        nx.draw_networkx_labels(graph, ax=ax, pos=layout, **params)


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


def highlight_subgraphs(graphs, colors, ax=None, label_nodes=True):
    if ax is None:
        ax = plt.gca()

    layout = graphviz_layout(graphs[0], prog='fdp')
    for graph, color in zip(graphs, colors):
        _draw_graph(graph, layout, ax, label_nodes, color)