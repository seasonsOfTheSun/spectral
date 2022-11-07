import utils
from info_measures import SingleInfo,PairInfo

gene_entropy = utils.load_entropy_df()
gene_df      = utils.load_gene_df()

selected_genes = gene_entropy.index[(gene_entropy["std"] > 0.3)]
n_genes = len(selected_genes)
tick = 0
N = (n_genes*(n_genes-1)/2)
for i,j in it.combinations(selected_genes,2):
  temp = {}
  pair = PairInfo(gene_df.iloc[:,i],gene_df.iloc[:,j])

  part1 = SingleInfo(pair.v1)
  part2 = SingleInfo(pair.v2)

  temp["indep_cross_to_normal"] = part1.cross_to_normal() + part2.cross_to_normal()
  temp["indep_entropy"] = part1.empirical_entropy() + part2.empirical_entropy()
  temp["cross_to_normal"] = pair.cross_to_normal()
  temp["empirical_entropy"] = pair.empirical_entropy()
  temp["gene1"] = gene_df.columns[i]
  temp["gene2"] = gene_df.columns[j]
  out[(i,j)] = temp

  tick += 1
  print(round(tick/N, 3))

pd.DataFrame(out).transpose().to_csv("data/intermediate/gene_entropy.csv")
