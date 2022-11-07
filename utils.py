import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

prefix = "data/"

def load_gene_df():
  df = pd.read_csv(prefix+"raw/CRISPR_gene_effect.csv", index_col = 0)
  print(f"remove {sum(df.isna().any(axis=1))} cell lines since they don't have measurements for all genes")
  df = df.loc[~df.isna().any(axis=1),:]
  hgnc = df.columns.map(lambda x:x.split(" ")[0])
  entrez = df.columns.map(lambda x:x.split(" ")[1][1:-1])
  df.columns = hgnc
  return df

def load_cell_line_df():
  cell_line_df = pd.read_csv(prefix+"raw/sample_info.csv")
  cell_line_df["tissue_type"] = cell_line_df.CCLE_Name.map(lambda x: ("N/A (HEKT)" if x is np.nan  else " ".join(x.split("_")[1:])))
  return cell_line_df

def load_entropy_df():
  return pd.read_csv(prefix+"raw/gene_entropy.csv", index_col=0)
