import utils
from info_measures import SingleInfo,PairInfo
import numpy as np
import pandas as pd
import itertools as it
import evectors

import evectors
gene_df = utils.load_gene_df()
G = evectors.umap_network(gene_df)
e,evecs = evectors.get_evecs(G)

gene_df = utils.load_gene_df()
q = np.quantile(gene_df.std(), 0.995)
selected_genes = gene_df.columns[gene_df.std() > q]
n_genes = len(selected_genes)

out = []
tick = 0
N = n_genes*evecs.shape[1]
for i,j in it.product(selected_genes,range(1,evecs.shape[1])):
    temp = {}
    pair = PairInfo(gene_df[i],evecs[:,j])

    part1 = SingleInfo(pair.v1)
    part2 = SingleInfo(pair.v2)

    temp["indep_cross_to_normal"] = part1.cross_to_normal() + part2.cross_to_normal()
    temp["indep_entropy"] = part1.empirical_entropy() + part2.empirical_entropy()
    temp["cross_to_normal"] = pair.cross_to_normal()
    temp["empirical_entropy"] = pair.empirical_entropy()
    temp["gene"] = i
    temp["evector_index"] = j
    temp["evector"] = f"Eigenvector-{j}"
    out.append(temp)

    tick += 1
    print(round(tick/N, 3))
entropy_evector_gene = pd.DataFrame(out)
entropy_evector_gene.to_csv("data/intermediate/entropy_evector_gene_pair.csv")
