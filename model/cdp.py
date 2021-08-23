import numpy as np
import gc


class Data:
    def __init__(self, name):
        self.name = name
        self.links = set()

    def add_link(self, other):
        self.links.add(other)
        other.links.add(self)


def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(
                    sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                valid = False
                remain.update(group)
                break
        if valid:  # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)
    # print("\tth: {}, remain: {}".format(th, len(remain)))
    return result, remain


def graph_propagation(edges, score, max_sz, step=0.1, max_iter=100):
    th = score.min()

    # construct graph
    score_dict = {}  # score lookup table
    for i, e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((np.max(nodes) + 1), dtype=np.int32)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l in link_idx:
        vertex[l[0]].add_link(vertex[l[1]])

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)

    # iteration
    components = comps[:]
    iter_num = 0
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(
            remain, max_sz, score_dict, th)
        components.extend(comps)
        iter_num += 1
        if iter_num >= max_iter:
            print("Warning: The iteration reaches max_iter: {}".format(max_iter))
            max_sz *= 2
    return components


def sample(nbrs, dist, th=0.7):
    simi = 1.0 - dist
    anchor = np.tile(np.arange(len(nbrs)).reshape(
        len(nbrs), 1), (1, nbrs.shape[1]))
    idx = np.where(simi > th)

    pairs = np.hstack((anchor[idx].reshape(-1, 1),
                       nbrs[idx].reshape(-1, 1)))
    scores = simi[idx]

    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores


def cdp(pairs, scores, max_sz, step, max_iter, n):
    components = graph_propagation(pairs, scores, max_sz, step, max_iter)

    # collect results
    cdp_res = []
    for c in components:
        cdp_res.append(sorted([n.name for n in c]))

    pred = -1 * np.ones(n, dtype=np.int32)
    for i, c in enumerate(cdp_res):
        pred[np.array(c)] = i

    valid = np.where(pred != -1)
    _, unique_idx = np.unique(pred[valid], return_index=True)
    pred_unique = pred[valid][np.sort(unique_idx)]
    pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
    pred_mapping[-1] = -1
    pred = np.array([pred_mapping[p] for p in pred])
    return pred


def main(knn_path, th, max_sz, step, max_iter):
    knns = np.load(knn_path)[..., 1:]
    nbrs, dists = knns[0].astype(int), knns[1]
    pairs, scores = sample(nbrs, dists, th=th)
    n = len(nbrs)
    del knns, nbrs, dists
    gc.collect()
    return cdp(pairs, scores, max_sz, step, max_iter, n)

