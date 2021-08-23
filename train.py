import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from dataset import FaceDataset, collate_sparse


def train(epochs, args, model):
    collate_fn = None if args.cdp else collate_sparse
    loader = DataLoader(FaceDataset(args), 512, num_workers=6, shuffle=True, collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=0.001)
    best_loss = 1e3
    model.train()
    for epoch in range(1, epochs + 1):

        running_loss = 0.0
        for data in tqdm(loader):
            data = [d.to("cuda") for d in data]
            optimizer.zero_grad()
            loss = model(*data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("epoch: {}, loss: {}".format(epoch, running_loss))
        if best_loss > running_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), "best.pth")
    print('Finished Training')
