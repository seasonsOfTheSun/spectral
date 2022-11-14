import pandas as pd
import numpy as np
import utils
from info_measures import SingleInfo
import pandas as pd
import utils


gene_df = utils.load_gene_df()
n_genes = gene_df.shape[1]
out = {}
for i in range(n_genes):
  print(round(i/n_genes,5))
  try:
    gene=SingleInfo(gene_df.iloc[:,i], n_bins=50)
    out[i]={}
    out[i]["name"] = gene_df.columns[i]
    out[i]["mean"]=gene.mean
    out[i]["std"]=gene.std
    out[i]["empirical_entropy"]=gene.empirical_entropy()
    out[i]["normal_entropy"]=gene.normal_entropy()
    out[i]["cross_to_normal"]=gene.cross_to_normal()
    out[i]["kl_div_to_normal"]=gene.kl_div_to_normal()
  except ValueError as e:
    print(i,e,hgnc[i])

gene_entropy = pd.DataFrame(out).transpose()
gene_entropy.sort_values("normal_entropy")
gene_entropy.to_csv(prefix + "raw/gene_entropy.csv")
