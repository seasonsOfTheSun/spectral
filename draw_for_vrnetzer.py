import networkx as nx
import pandas as pd
import numpy as np
import sphere

import sys 
attempt = sys.argv[1]

df = pd.read_csv("data/intermediate/gene_evector_entropy.csv", index_col=[0,1])
df["kl"] = df["cross_to_normal"] - df["empirical_entropy"]

G = nx.Graph()
for i,v in df["kl"].iteritems():
    evector,gene = i
    if not np.isnan(v):
        G.add_edge(evector, gene, weight=np.exp(-v))

pos = sphere.spherical_spring_layout(G)
pos_df = pd.DataFrame(pos).T

for i in range(3):
    pos_df[i] = (pos_df[i] - (-1.01))/2.02

#for i in range(3):
#    pos_df[i] = (pos_df[i] - pos_df[i].min())/(pos_df[i].max() - pos_df[i].min())

e_or_g = pos_df.index.map(lambda x: type(x)==np.int)
pos_df["type"] = np.where(e_or_g, "Eigenvector", "Gene")

e_color = [150,150,0]
g_color = [0,0,255]
pos_df["r"] = np.where(e_or_g, e_color[0], g_color[0])
pos_df["g"] = np.where(e_or_g, e_color[1], g_color[1])
pos_df["b"] = np.where(e_or_g, e_color[2], g_color[2])
pos_df["alpha"] = 100

pos_df = pos_df[[0,1,2,"r","g","b","alpha","type"]]

pos_df.to_csv(f"for_vrnetzer/node_attempt_{attempt}.csv", header=None, index=None)

convert_label = {v:i for i,v in enumerate(pos_df.index)}
link_df = pd.DataFrame(G.edges()).applymap(convert_label.get)
link_df.columns=["source", "target"]

link_df["r"] = 255#np.where
link_df["g"] = 0#np.where
link_df["b"] = 0#np.where
link_df["alpha"] = 50#np.where

link_df.to_csv(f"for_vrnetzer/link_attempt_{attempt}.csv", header=None, index=None)
