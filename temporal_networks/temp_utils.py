"""
Useful functions to work with temporal networks
to complement packages such as networkx, pathpy, and teneto
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

import pathpy as pp
import teneto as tnt

import seaborn as sb
from matplotlib import animation
from networkx.drawing.nx_agraph import graphviz_layout


#==========================
# Short utility functions for networkx
#==========================

def nx_cbar(cmap, vmin, vmax, label=None, shrink=1, ticks=None, ax=None):

	"""
	Add colorbar to networkx graph plot,
	e.g. when nodes are coloured according to degree
	"""
    
	if ax==None:
		ax = plt.gca()
        
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
	sm._A = []
	clb = plt.colorbar(sm, ax=ax, shrink=shrink, ticks=ticks)
	clb.ax.set_title(label)


	return sm


#=========================
# FUNCTIONS FOR TENETO
#=========================

def aggregate_adj(adj_matrices, t_ax=0):
    
    """
    Return aggregate network of networks
    
    Input
    -----
    network: list or array
        list of adjacency matrices, dimension (T,N,N)
        
    Output
    ------
    agg_net: numpy array
        adjacency matrix of aggregate network, dimension (N,N)
    """
    
#     if isinstance(networks, list) :
#         networks = np.array(networks)
        
#     print(networks.shape)
    # check that time is zeroth axis
#     assert networks.shape[0] != networks.shape[1]
    
    # sum on time axis
    sum_adj = np.sum(adj_matrices, axis=t_ax)
    # binarise
    agg_adj = sum_adj
    agg_adj[agg_adj > 0] = 1
    
    return agg_adj


def time_partition(tnet, t_window) : 
    
    """
    Return temporal network obtained by aggregating original temporal network
    inside each consecutive time window containing t_window time points.
    
    INPUT
    -----
    tnet: teneto TemporalNetwork
        original temporal network
    
    t_window: int
        length of time window in time points
    
    OUTPUT
    ------
    tnet_agg: TemporalNetwork
    
    """
    
    if tnet.T % t_window != 0 :
        print('warning: tnet.T not divisible by t_window')
        T_agg = tnet.T // t_window + 1
        # and then, the last window aggregates only tnet.T % t_window points, i.e. < T_agg
    else : 
        T_agg = tnet.T // t_window

    if isinstance(tnet.network, pd.core.frame.DataFrame) :
        snapshots = tnet.df_to_array()
    elif isinstance(tnet.network, np.ndarray) :
        snapshots = tnet.network
    else : 
        snapshots = []
        print('wrong tnet.network type')
        
    # put time as zeroth axis
    snapshots = np.swapaxes(snapshots, 0, 2)
    snapshots_agg = np.zeros((T_agg, tnet.N, tnet.N))
    
    for i in range(T_agg) : 
        
        i_window = t_window * i # start index of window

        snapshots_agg[i] = aggregate_adj(snapshots[i_window:i_window+t_window])
    
    # swich time back to last axis
    snapshots_agg = np.swapaxes(snapshots_agg, 0, 2)
            
    tnet_agg = tnt.TemporalNetwork(from_array=snapshots_agg) 
        
    return tnet_agg

# VISUALISATION

# def animate_tempnet(tnet) :

# 	# make sure format is array of networks 
#     if isinstance(tnet.network, pd.core.frame.DataFrame) :
#         snapshots = tnet.df_to_array()
#     elif isinstance(tnet.network, np.ndarray) :
#         snapshots = tnet.network
#     else : 
#         snapshots = []
#         print('wrong tnet.network type')

#     # set layout position from aggregated network (in nx)
#     # snapshots = np.swapaxes(snapshots, 0, 2) # put time as zeroth axis
#     agg_adj = aggregate_adj(snapshots, t_ax=2)
#     G = nx.Graph(agg_adj)	
#     pos = graphviz_layout(G)

#     # draw fixed elements
#     fig, ax = plt.subplots(figsize=(12,10))

#     # define animated elements
#     def animate(i):

# 	    
