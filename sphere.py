import networkx as nx

import numpy as np
import scipy as sp
import scipy.sparse  # call as sp.sparse

def spherical_spring_layout(G,
       k=None,
       pos=None,
       fixed=None,
       iterations=1,
       threshold=1e-4,dim=3,
       target_radius = 1,
       sphere_to_spring = 2, 
       seed = None):

    if seed == None:
        seed = nx.utils.create_random_state() 

    nodelist = list(G.nodes())
    A= nx.adjacency_matrix(G, nodelist=nodelist) 
    A = A.astype('float64')

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except AttributeError:
        A = (sp.sparse.coo_array(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.randn(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []



    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    t_sphere = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t/(iterations+1)
    dt_sphere = (t_sphere/(iterations+1)) / sphere_to_spring

    displacement = np.zeros((dim, nnodes))
    displacement_sphere = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        displacement_sphere *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue

            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = A.getrowview(i).toarray()  # TODO: revisit w/ sparse 1D container
            # displacement "force"
            displacement[:, i] += (delta * (k*k/distance**2 - Ai*distance/k)).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T



#        displacement_sphere = pos
#        length_sphere = np.sqrt((displacement_sphere**2).sum(axis=1))
#        sign_sphere = np.where(length_sphere > target_radius, -1, 1)
#        delta_pos_sphere = (sign_sphere*displacement_sphere.T * t_sphere / length_sphere).T

        pos += delta_pos
#        pos += delta_pos_sphere
        # cool temperature
        t -= dt
#        t_sphere -= dt_sphere
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break

    # project onto sphere surface
    pos = (pos.T/np.linalg.norm(pos, axis=1)).T

    pos = dict(zip(nodelist, pos))
    return pos


# (inverse) ratio of sphere effect decrese with temperature to spring effect decrese with temperature
G = nx.erdos_renyi_graph(100, 0.1) 




