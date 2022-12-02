


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_mi(mi):
    fig,ax = plt.subplots()
    ax.hist(mi.flatten(), bins = 40, color= 'k')
    ax.set_xlabel("Mutual information")
    ax.set_ylabel("freq.")
    fig_hist = fig

    fig,ax = plt.subplots()
    h = ax.imshow(mi[::-1,::-1])
    cbar = fig.colorbar(h)
    ax.set_ylabel("Eigenvector")
    ax.set_xlabel("Eigenvector")
    cbar.set_label("Mutual Information")
    fig_heat = fig
    return fig_hist,fig_heat


def pretty_draw_network(G,pos=None):
    if pos == None:
        pos = nx.kamada_kawai_layout(G)
    fig,ax= plt.subplots(figsize=[10,10])
    nx.draw_networkx_nodes(G,ax=ax, pos=pos,node_color='white',edgecolors='k')
    w = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, ax=ax, pos=pos,width=np.array(list(w.values())))
    ax.axis('off')

def summarize_pruning(edge_list, w_list, sp_list):
    fig,ax = plt.subplots()
    ax.hist(w_list - sp_list, bins=200, color='k');
    ax.set_xlabel("Outranking by shortest path")
    ax.set_ylabel("freq")

    fig,ax = plt.subplots()
    ax.scatter(w_list, sp_list, c='k', s=3)
    ax.set_ylabel("shortest path length between end-nodes")
    ax.set_xlabel("edge weight")
    ax.set_title("For each edge in MI network:")

    fig,ax = plt.subplots()
    ax.scatter(w_list, sp_list, c=(w_list-sp_list)<0.01, s=3, cmap = 'seismic')
    ax.set_ylabel("shortest path length between end-nodes")
    ax.set_xlabel("edge weight")
    ax.set_title("isolating redundant edges")
