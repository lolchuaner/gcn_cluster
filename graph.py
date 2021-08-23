import numpy as np
import scipy.sparse as sp
from sklearn.neighbors._base import _radius_neighbors_from_graph
from sklearn.cluster import DBSCAN


def knn2adj(nbrs, dists):
    n = len(nbrs)
    row, col = np.where(dists < 0.5)
    data = dists[row, col]
    col = nbrs[row, col]
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    return adj


def node2adj(nbrs, nodes):
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    nodes_map = {j: i for i, j in enumerate(nodes)}

    for i, nbr in enumerate(nbrs[nodes]):
        for j in nbr:
            if j in nodes_map:
                adj[i, nodes_map[j]] = 1
    return adj


def graph2label(graphs, n):
    pred = np.full(n, -1, dtype=np.intp)
    for i, g in enumerate(graphs):
        pred[g] = i
    return solve_noise(pred)


def solve_noise(pred):
    label_num = np.max(pred) + 1

    for i in range(len(pred)):
        if pred[i] == -1:
            pred[i] = label_num
            label_num += 1
    return pred


def create_graph(nbrs, dists, r, min_samples, labels):
    adj = knn2adj(nbrs, dists)
    pred = DBSCAN(r, min_samples, metric="precomputed").fit_predict(adj)
    n = np.max(pred) + 1
    graphs = [[] for _ in range(n)]

    for i, j in enumerate(pred):
        if j != -1:
            graphs[j].append(i)

    neighborhoods = _radius_neighbors_from_graph(adj, r + 0.15, return_distance=False)
    new_graphs, y, feats = [], [], []
    for g in graphs:
        l, c = np.unique(labels[g], return_counts=True)
        nodes = np.unique(np.concatenate(neighborhoods[g]))
        nodes = np.concatenate([g, np.setdiff1d(nodes, g)])
        new_graphs.append(nodes)
        x = np.zeros((len(nodes), 2), dtype=np.float32)
        x[:len(g), 0] = 1
        x[len(g):, 1] = 1
        feats.append(x)
        y.append(labels[nodes] == l[np.argmax(c)])
    return new_graphs, feats, y


def fps(nbrs, labels, k):
    v = []
    n = len(nbrs)
    points = np.ones(n)
    for i in range(n):
        if points[i] == 1:
            v.append(i)
            points[nbrs[i, :k]] = 0
    graphs = nbrs[v]
    y = [labels[g] == labels[g[0]] for g in graphs]
    x = np.zeros((80, 2), dtype=np.float32)
    x[:1, 0] = 1
    x[1:, 1] = 1
    return graphs, x, y
