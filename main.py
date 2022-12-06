from info_measures import SingleInfo, PairInfo
import networkx as nx
import umap
import numpy as np
import pandas as pd
import scipy.sparse
import itertools as it

def umap_network(X, nodes=None, nneighbors = 10, metric = 'euclidean'):
    rndstate = np.random.RandomState(nneighbors)
    if nodes == None:
        nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X, 10, rndstate, metric)
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(nodes)).get)

def get_evecs(G, n_evectors = 50):
  
  nodelist = np.array(G.nodes())
  # rescale rows and columns by degree
  normalized_adjacency = scipy.sparse.eye(G.order()) - nx.normalized_laplacian_matrix(G)
  e,evecs = scipy.sparse.linalg.eigsh(normalized_adjacency, k = n_evectors)
  e = e[::-1]
  evecs = evecs[:,::-1]
  return e,evecs

def calculate_information(evecs,n_evectors=None):

  if n_evectors == None:
    n_evectors = evecs.shape[1]
  mi = np.zeros((n_evectors,n_evectors))
  kl = np.zeros((n_evectors,n_evectors))

  for i,j in it.product(range(n_evectors), range(n_evectors)):
    if i == j:
      mi[i,j] = np.nan
      kl[i,j] = np.nan
    else:
      # pair Class that calculates numerous information related
      # stats for the two eigenvectors together
      pair = PairInfo(evecs[:,i],evecs[:,j])
      # calcualte similarity (KL divergennce) to normal of their joint distributions
      kl[i,j] = pair.cross_to_normal() - pair.empirical_entropy()
      # calcualte mutual information betweenn pairs of eigenvectors
      independent_entropy = pair.part1.empirical_entropy() + pair.part2.empirical_entropy()
      mi[i,j] = independent_entropy - pair.empirical_entropy()
  return mi, kl

def mi_network(mi):
  G = nx.from_numpy_array(mi.clip(0)) # clip negative edge weights since they are due to approximation errors
  for i in G.nodes():
    G.remove_edge(i,i)
  return G


def transitivize_mi_network(mi_G):
    """ this is it. """
    d_lengths = dict(nx.shortest_paths.all_pairs_dijkstra_path_length(mi_G,weight='distance'))
    d_paths = dict(nx.shortest_paths.all_pairs_dijkstra_path(mi_G,weight='distance'))
    pruned_G = mi_G.copy()
    for u,v in mi_G.edges():
        # print("all intermediate steps must be more than:", mi_G.edges()[(u,v)]['mutual_information'])
        single_step = mi_G.edges()[(u,v)]['mutual_information']
        # print("all intermediate steps must be more than:",-np.log(mi_G.edges()[(u,v)]['distance'])/n)
        path = d_paths[u][v]

        delete=True
        for i,j in zip(path[:-1],path[1:]):
            # print(mi_G.edges()[(i,j)]['mutual_information'])
            tol = 0.00001 # avoid floating point nonsense
            if single_step >= mi_G.edges()[(i,j)]['mutual_information']-tol:
                delete=False
        if delete:
            # print("deleting")
            pruned_G.remove_edge(u,v)
        else:
            print("not deleting")
    return pruned_G
