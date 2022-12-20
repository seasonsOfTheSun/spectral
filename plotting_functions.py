import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats
import os

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


def plot_features(xy,X,dataset_name, dataset_feature_names):
    os.makedirs(f"plots/{dataset_name}/features/", exist_ok=True) 
    fig,axes = plt.subplots(nrows=6,ncols=3,figsize=[15,20])
    for i,ax in enumerate(axes.flatten()):
        try:
            c = ax.scatter(xy[:,0], xy[:,1], s = 20, c = X[:,i],cmap="viridis",edgecolor='k',linewidth=0.5)
            cax = plt.colorbar(c,ax=ax)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            cax.set_label(dataset_feature_names[i]+"\n(normalized)")
        except IndexError:
            ax.axis("off")
    fig.savefig(f"plots/{dataset_name}/features_all.svg")


def plot_evecs(xy,evecs,dataset_name):
    os.makedirs(f"plots/{dataset_name}/",exist_ok=True)
    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=[16,10])
    for i,ax in enumerate(axes.flatten()):

        c = ax.scatter(xy[:,0], xy[:,1], s = 20, c = evecs[:,i+1],cmap="PiYG",edgecolor='k',linewidth=0.5)
        cax = plt.colorbar(c,ax=ax)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cax.set_label("SEV-"+str(i+1))
    fig.savefig(f"plots/{dataset_name}/SEV_all.svg") 
