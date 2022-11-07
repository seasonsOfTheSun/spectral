from info_measures import SingleInfo, PairInfo
from utils import

import networkx as nx
import umap
import numpy as np
import scipy.sparse

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

G = umap_network(gene_df)
e,evecs = get_evecs(G)

tick = 0
n_genes = len(gene_df.columns)
n_evectors = evecs.shape[1]
N = n_genes*n_evectors
out = {}
import itertools as it
for i in range(n_evectors):
  for j in range(n_genes):
    pair = PairInfo(gene_df.iloc[:,j],evecs[:,i])

    part1 = SingleInfo(pair.v1)
    part2 = SingleInfo(pair.v2)

    temp = {}
    temp["indep_cross_to_normal"] = part1.cross_to_normal() + part2.cross_to_normal()
    temp["indep_entropy"] = part1.empirical_entropy() + part2.empirical_entropy()
    temp["cross_to_normal"] = pair.cross_to_normal()
    temp["empirical_entropy"] = pair.empirical_entropy()
    temp["gene"] = gene_df.columns[j]
    temp["evec"] = i

    out[(i,j)] = temp

    tick += 1
    print(round(tick/N, 3), end="\r")
