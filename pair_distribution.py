import utils
from info_measures import SingleInfo,PairInfo
import numpy as np
import pandas as pd
import itertools as it

gene_df = utils.load_gene_df()
q = np.quantile(gene_df.std(), 0.995)
selected_genes = gene_df.columns[gene_df.std() > q]
n_genes = len(selected_genes)

out = []
tick = 0
N = (n_genes*(n_genes-1)/2)
for i,j in it.combinations(selected_genes,2):
  temp = {}
  pair = PairInfo(gene_df[i],gene_df[j])

  part1 = SingleInfo(pair.v1)
  part2 = SingleInfo(pair.v2)

  temp["indep_cross_to_normal"] = part1.cross_to_normal() + part2.cross_to_normal()
  temp["indep_entropy"] = part1.empirical_entropy() + part2.empirical_entropy()
  temp["cross_to_normal"] = pair.cross_to_normal()
  temp["empirical_entropy"] = pair.empirical_entropy()
  temp["gene1"] = i
  temp["gene2"] = j
  out.append(temp)

  tick += 1
  print(round(tick/N, 3))

pd.DataFrame(out).transpose().to_csv("data/intermediate/gene_pair_entropy.csv", index=None).T
