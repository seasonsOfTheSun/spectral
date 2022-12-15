from info_measures import SingleInfo, PairInfo
import networkx as nx
import umap
import numpy as np
import pandas as pd
import scipy.sparse
import itertools as it
import tqdm


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


def evec_info_df(evecs):
    out = []
    import tqdm
    import info_measures
    import itertools as it
    # maybe capture som more standard relationship measuures to see how they differ
    total  = evecs.shape[1]*(evecs.shape[1]-1)/2

    for i,j in tqdm.tqdm(it.combinations(range(evecs.shape[1]), r=2),total=total):
        pair = PairInfo(evecs[:,i], evecs[:,j])
        temp = pair.collect()

        temp["eigenvector_name_1"] = "SEV-"+str(i)
        temp["eigenvector_name_2"] = "SEV-"+str(j)
        temp["eigenvector_index_1"] = i
        temp["eigenvector_index_2"] = j
        out.append(temp)
    import pandas as pd
    df = pd.DataFrame(out)
    df= df[df.eigenvector_name_1 != "SEV-0"]
    df["kld_to_normal"] = df.cross_to_normal - df.empirical_entropy
    df["mutual_information"] = df.individual_entropy - df.empirical_entropy
    return df

def evec_feature_info_df(evecs, X, feature_names):
    out = []

    # maybe capture som more standard relationship measuures to see how they differ
    total  = evecs.shape[1]*X.shape[1]
    iterable = list()
    for i,j in tqdm.tqdm(it.product(range(evecs.shape[1]), range(X.shape[1])),total=total):
        pair = PairInfo(evecs[:,i], X[:,j])
        temp = pair.collect()

        temp["eigenvector_name"] = "SEV-"+str(i)
        temp["feature_name"] = feature_names[j]
        temp["eigenvector_index"] = i
        temp["feature_index"] = j
        out.append(temp)
    feature_df = pd.DataFrame(out)

    feature_df= feature_df[feature_df.eigenvector_name != "SEV-0"]
    feature_df["kld_to_normal"] = feature_df.cross_to_normal - feature_df.empirical_entropy
    feature_df["mutual_information"] = feature_df.individual_entropy - feature_df.empirical_entropy
    return feature_df

def pivot_evec_info(df,column="mutual_information"):
    mi = pd.pivot(df[["eigenvector_name_1",
                      "eigenvector_name_2",
                      column]],
             index = "eigenvector_name_1",
             columns = "eigenvector_name_2",
             values="mutual_information"
            )
    evec_names = sorted(set(df["eigenvector_name_1"]) | set(df["eigenvector_name_2"]), key=lambda x : int(x.split("-")[-1]))
    mi = mi.reindex(evec_names).T.reindex(evec_names)
    assert all(mi.T.isna() | mi.isna())
    mi = mi.fillna(0) + mi.T.fillna(0)
    for i in range(mi.shape[0]):
        mi.iloc[i,i] = np.nan
    return mi

def pivot_evec_feature_info(feature_df, column="mutual_information"):
    feature_mi = pd.pivot(feature_df[["eigenvector_name",
                              "feature_name",
                              column]],
             index = "eigenvector_name",
             columns = "feature_name",
             values="mutual_information"
            )
    return feature_mi

def mi_network(mi,nodes,n = 100):
    
    mi_G = nx.from_numpy_array(mi.clip(0)) # clip negative edge weights since they are due to approximation errors                                       
    for i in mi_G.nodes():
        mi_G.remove_edge(i,i)
    
    for u,v in mi_G.edges():
        mi_G.edges()[(u,v)]['mutual_information'] = mi_G.edges()[(u,v)]['weight']
        mi_G.edges()[(u,v)]['distance'] = np.exp(-n*mi_G.edges()[(u,v)]['weight'])
    
    mi_G = nx.relabel_nodes(mi_G, dict(enumerate(nodes)).get)
    return mi_G

def verbose_pruning(mi_G):
    tol = 0.001 # tol is there to avoid floating point nonsense or maybe provide a useful role not sure yet
    d_lengths = dict(nx.shortest_paths.all_pairs_dijkstra_path_length(mi_G,weight='distance'))
    d_paths = dict(nx.shortest_paths.all_pairs_dijkstra_path(mi_G,weight='distance'))
    pruned_G = mi_G.copy()

    out = []
    for u,v in mi_G.edges():
        single_step = mi_G.edges()[(u,v)]['mutual_information']
        path = d_paths[u][v]

        delete=True
        for i,j in zip(path[:-1],path[1:]):
            # print(mi_G.edges()[(i,j)]['mutual_information'])
            # tol is there toavoid floating point nonsense or maybe provide a useful role not sure yet
            if single_step >= mi_G.edges()[(i,j)]['mutual_information']-tol:
                delete=False
        if delete:
            pruned_G.remove_edge(u,v)
        min_in_path = min([mi_G.edges()[(i,j)]['mutual_information'] for i,j in zip(path[:-1],path[1:])])
        out.append([single_step, min_in_path, delete])
    return out

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


def full_analysis(X, feature_names):
    X = (X-X.mean(axis=0))/X.std(axis=0)

    index = [str(i) for i in range(X.shape[0])]
    G = main.umap_network(X, nodes=index)

    e,evecs = main.get_evecs(G, n_evectors = 20)

    feature_df = main.evec_feature_info_df(evecs, X, feature_names)
    df = main.evec_info_df(evecs)
    evec_mi = main.pivot_evec_info(df);
    feature_mi = main.pivot_evec_feature_info(feature_df);

    info_G = mi_network(evec_mi.values,evec_mi.columns)
    main.transitivize_mi_network(info_G)

