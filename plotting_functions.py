import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats

def plot_normal(ax, mean, std, min_, max_):
  xrange = np.linspace(min_, max_, 50)
  log_p = -0.5*((xrange - mean)/std)**2  - np.log(std) - 0.5*np.log(2*np.pi)
  ax.plot(xrange, np.exp(log_p), c = 'r')

def plot_hist_normal(ax,v,label):
  ax.hist(v, bins=40,density=True,color="#555555");
  std = np.std(v)
  mean = np.mean(v)
  min_,max_ = v.min(), v.max()
  plot_normal(ax, mean, std, min_, max_)
  ax.set_xlabel(label)
  ax.set_ylabel("freq")

def draw_ellipse(pair,ax,r = 1,color='r'):
    # Covariance Recovered from Samples
    t = np.linspace(0, 2*np.pi, 100)
    xy_ellipse = np.vstack([pair.xscale*r*np.cos(t),pair.yscale*r*np.sin(t)]).T
    xy_ellipse = np.matmul(pair.rot, xy_ellipse.T).T
    ax.plot(pair.mean[0]+xy_ellipse[:,0] ,pair.mean[1]+xy_ellipse[:,1], c=color)

def plot_scatter_normal(ax,pair):
    ax.scatter(pair.v1,pair.v2,c="#555555")
    for t in np.linspace(0,1,5):
      pair.draw_ellipse(ax,r = scipy.stats.chi2(df=2).isf(t))
    ax.axis('equal')

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
