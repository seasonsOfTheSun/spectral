from info_measures import SingleInfo, PairInfo
import networkx as nx
import umap
import numpy as np
import pandas as pd
import scipy.sparse
import itertools as it

def umap_network(X, nneighbors = 10, metric = 'euclidean'):
    rndstate = np.random.RandomState(nneighbors)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X, 10, rndstate, metric)
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

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

def identify_redundant_edges(G,edge_list=None):

  if edge_list == None:
    edge_list = list(G.edges())

  sp_list = []
  w_list = []
  sp_dict = dict(nx.shortest_path_length(G, weight='weight'))
  
  for i,j in edge_list:
    w_list.append(G.edges()[(i,j)]['weight'])
    sp_list.append(sp_dict[i][j])
  sp_list = np.array(sp_list)
  w_list = np.array(w_list)
  return edge_list, w_list, sp_list

def transitivize_mi_network(G,edge_list, w_list, sp_list):
  pruned_G = G.copy()
  tol = 0.01 # set some tolerance to avoid floating precision issues
  for i,uv in enumerate(edge_list):
    u,v = uv
    w_list, sp_list
    if w_list[i] - sp_list[i] > tol:
      pruned_G.remove_edge(u,v)
  return pruned_G



