import numpy as np


def create_neighbor_lookup(nbrs):
    nn_lookup = {}
    for i in range(nbrs.shape[0]):
        nn_lookup[i] = nbrs[i, :]
    return nn_lookup


def aro_clustering(nbrs, dists, thresh):
    # Clustering
    clusters = []
    # Start with the first face
    nodes = set(list(np.arange(0, dists.shape[0])))
    plausible_neighbors = create_plausible_neighbor_lookup(nbrs, dists, thresh)
    while nodes:
        # Get a node
        n = nodes.pop()
        # This contains the set of connected nodes
        group = {n}
        # Build a queue with this node in it
        queue = [n]
        # Iterate over the queue
        while queue:
            n = queue.pop(0)
            neighbors = plausible_neighbors[n]
            # Remove neighbors we've already visited
            neighbors = nodes.intersection(neighbors)
            neighbors.difference_update(group)

            # Remove nodes from the global set
            nodes.difference_update(neighbors)

            # Add the connected neighbors
            group.update(neighbors)

            # Add the neighbors to the queue to visit them next
            queue.extend(neighbors)
        # Add the group to the list of groups
        clusters.append(group)
    return clusters


def create_plausible_neighbor_lookup(nbrs, dists, thresh):
    n_vectors = nbrs.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        plausible_neighbors[i] = set(
            list(nbrs[i, np.where(dists[i, :] <= thresh)][0]))
    return plausible_neighbors


def clusters2labels(clusters, num):
    labels_ = -1 * np.ones(num, dtype=np.int)
    for lb, c in enumerate(clusters):
        idx = np.array([int(x) for x in list(c)])
        labels_[idx] = lb
    return labels_


def aro(knn_path, th_sim):
    knns = np.load(knn_path)[:, :, 1:]
    nbrs, dists = knns[0].astype(int), knns[1]
    clusters = aro_clustering(nbrs, dists, 1. - th_sim)
    labels_ = clusters2labels(clusters, len(nbrs))
    return labels_
