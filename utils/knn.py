from tqdm import tqdm
import numpy as np
import faiss


def create_index(target):
    _, d = target.shape

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(target)
    return index


def batch_search(index, query, k, bs=int(1e4)):
    n = len(query)
    dists = np.zeros((n, k), dtype=np.float32)
    nbrs = np.zeros((n, k), dtype=np.int64)

    for sid in tqdm(range(0, n, bs), desc="faiss searching..."):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)
    return dists, nbrs


def faiss_search_knn(feat, k):
    index = create_index(feat)
    dists, nbrs = batch_search(index, feat, k=k)

    dists = 1. - dists
    dists[dists < 0] = 0
    knn = np.zeros((2, len(dists), k), dtype=np.float32)
    knn[0] = nbrs
    knn[1] = dists
    return knn


def main(base_path):
    for feat_name in ("train", "test"):
        feat = np.load("{}/{}_fea.npy".format(base_path, feat_name), allow_pickle=True)
        faiss.normalize_L2(feat)

        knn = faiss_search_knn(feat, k=80)

        np.save("{}/knns/{}_k_80.npy".format(base_path, feat_name), knn)
