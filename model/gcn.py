import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


def MLP(in_channels, channels):
    return nn.Sequential(
        nn.Linear(in_channels, channels),
        nn.ReLU(),
        nn.Linear(channels, channels),
        nn.ReLU()
    )


class GraphConv(nn.Module):
    def __init__(self, channels):
        super(GraphConv, self).__init__()
        self.lin1 = MLP(channels, channels)
        self.lin2 = MLP(channels, channels)

    def forward(self, x, adj):
        x0 = x

        if x.dim() == 2:
            x = torch.spmm(adj, x)
        else:
            x = torch.bmm(adj, x)
        x = self.lin1(x)

        x = self.lin2(x0 + x)
        return x


class Net(nn.Module):
    def __init__(self, channels):
        super(Net, self).__init__()
        self.lin0 = MLP(2, channels)
        self.convs = nn.ModuleList([
            GraphConv(channels) for _ in range(5)
        ])
        self.lin = nn.Linear(channels, 2)
        self.loss_func = sigmoid_focal_loss

    def forward(self, x, adj, y, batch=None):

        if batch is not None:
            n = len(x)
            data = torch.ones(len(adj[0])).cuda()
            adj = torch.sparse_coo_tensor(adj, data, (n, n))
        x = self.lin0(x)
        for conv in self.convs:
            x = conv(x, adj)

        x = self.lin(x)

        if self.training:
            y = y.flatten()
            y = F.one_hot(y, num_classes=2).float()
            return self.loss_func(x.view(-1, 2), y)

        x = torch.sigmoid(x)
        if batch is not None:
            out = []
            x = x[:, 1]
            for i in range(len(batch) - 1):
                out.append(x[batch[i]:batch[i + 1]].cpu().numpy())
            return out
        return x.cpu().numpy()
