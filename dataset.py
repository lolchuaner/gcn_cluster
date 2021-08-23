import numpy as np
import torch
from torch.utils.data import Dataset
from graph import create_graph, node2adj, fps


class FaceDataset(Dataset):
    def __init__(self, args):
        labels = np.load("{}/{}_label.npy".format(args.path, args.data))
        knn = np.load("{}/knns/{}_k_80.npy".format(args.path, args.data))
        nbrs, dists = knn[0].astype(int), knn[1]
        dists[dists < 0] = 0
        if args.cdp:
            self.graphs, self.feats, self.y = fps(nbrs, labels, 10)
        else:
            self.graphs, self.feats, self.y = create_graph(nbrs, dists, args.r, args.min_sample, labels)

        self.labels = labels
        self.nbrs = nbrs[:, 1:30]
        self.cdp = args.cdp
        self.dists = dists

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        nodes = self.graphs[idx]
        if self.cdp:
            x = torch.from_numpy(self.feats)
        else:
            x = torch.from_numpy(self.feats[idx])
        adj = node2adj(self.nbrs, nodes)
        adj = torch.from_numpy(adj)
        y = self.y[idx]
        y = torch.LongTensor(y)
        return x, adj, y


def collate_sparse(batch):
    x, adj, y = zip(*batch)

    x = torch.cat(x)
    y = torch.cat(y)

    cum = 0
    source, target, batch = [], [], [0]
    for a in adj:
        row, col = torch.where(a == 1)
        source.append(row + cum)
        target.append(col + cum)
        cum += len(a)
        batch.append(cum)
    batch = torch.LongTensor(batch)
    indices = torch.stack([torch.cat(source), torch.cat(target)])
    return x, indices, y, batch
