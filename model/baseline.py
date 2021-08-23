import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from graph import knn2adj


def kmeans(feat_path, n_clusters):
    feats = np.load(feat_path)
    model = MiniBatchKMeans(n_clusters, max_iter=100).fit(feats)
    return model.labels_


def spectral(knn_path, n_clusters):
    adj = create_adj(knn_path)
    adj.data = np.ones(len(adj.data))
    model = SpectralClustering(
        n_clusters,
        affinity="precomputed_nearest_neighbors").fit(adj)
    return model.labels_


def hac(feat_path, knn_path, n_clusters):
    feats = np.load(feat_path)
    adj = create_adj(knn_path)
    adj.data = np.ones(len(adj.data))
    model = AgglomerativeClustering(n_clusters,
                                    connectivity=adj).fit(feats)
    return model.labels_


def dbscan(knn_path):
    adj = create_adj(knn_path)
    model = DBSCAN(eps=0.3, min_samples=5,
                   metric="precomputed").fit(adj)
    return model.labels_


def create_adj(knn_path):
    knns = np.load(knn_path)[..., 1:]
    nbrs, dists = knns[0].astype(int), knns[1]
    adj = knn2adj(nbrs, dists)
    return adj
