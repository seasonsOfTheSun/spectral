from info_measures import SingleInfo, PairInfo
from utils import *

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

if __name__ == '__main__':

    gene_df = load_gene_df()
    G = umap_network(gene_df)
    e,evecs = get_evecs(G)

    pd.DataFrame(evecs, index=gene_df.index).to_csv("data/intermediate/evectors.csv")
    pd.Series(e).to_csv("data/intermediate/evalues.csv")

    q = np.quantile(gene_df.std(), 0.995)
    genes = gene_df.columns[gene_df.std() > q]
    n_genes = len(genes)
    n_evectors = evecs.shape[1]
    N = n_genes*n_evectors
    out = {}

    tick = 0
    for i in range(n_evectors):
      for j in genes:
        pair = PairInfo(gene_df.loc[:,j],evecs[:,i])

        part1 = SingleInfo(pair.v1)
        part2 = SingleInfo(pair.v2)

        temp = {}
        temp["indep_cross_to_normal"] = part1.cross_to_normal() + part2.cross_to_normal()
        temp["indep_entropy"] = part1.empirical_entropy() + part2.empirical_entropy()
        temp["cross_to_normal"] = pair.cross_to_normal()
        temp["empirical_entropy"] = pair.empirical_entropy()
        temp["gene"] = j
        temp["evec"] = i

        out[(i,j)] = temp

        tick += 1
        print(round(tick/N, 3), end="\r")

    evec_df = pd.DataFrame(out)
    evec_df.T.to_csv("data/intermediate/gene_evector_entropy.csv")
