import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from dataset import FaceDataset, collate_sparse
from utils.metrics import nmi, bcubed
from model import cdp
from graph import graph2label
import gc


def test_gcn_cdp(model, args):
    t0 = time.time()
    loader = DataLoader(FaceDataset(args), 512, num_workers=6)

    model.eval()
    scores = []
    for data in tqdm(loader):
        data = [d.to("cuda") for d in data]
        with torch.no_grad():
            pred = model(*data)
        scores.append(pred[:, 1:, 1])

    scores = np.concatenate(scores)
    graphs = loader.dataset.graphs
    labels = loader.dataset.labels
    sims = 1 - loader.dataset.dists[:, 1:]
    del loader
    gc.collect()
    v, nbr = graphs[:, 0], graphs[:, 1:]
    row, col = np.where(scores > 0.1)
    scores = sims[row, col]
    pairs = np.stack([v[row], nbr[row, col]], axis=-1)

    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    pred = cdp(pairs, scores, args.max_size, args.step, 150, len(labels))
    num = pred.max() + 1
    for i in range(len(pred)):
        if pred[i] == -1:
            pred[i] = num
            num += 1
    print(time.time() - t0)
    print(nmi(labels, pred))
    print(bcubed(labels, pred))


def test_gcn_dbscan(model, args):
    t0 = time.time()
    loader = DataLoader(FaceDataset(args), 512, num_workers=6, collate_fn=collate_sparse)

    model.eval()
    scores = []
    for data in tqdm(loader):
        data = [d.to("cuda") for d in data]
        with torch.no_grad():
            pred = model(*data)
        scores += pred

    graphs = loader.dataset.graphs
    labels = loader.dataset.labels
    graphs = [g[s > 0.5] for g, s in zip(graphs, scores)]
    pred = graph2label(graphs, len(labels))
    print(time.time() - t0)
    print(nmi(labels, pred))
    print(bcubed(labels, pred))
