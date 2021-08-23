import argparse
from model import Net
import torch
from train import train
from test import test_gcn_cdp, test_gcn_dbscan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cdp', action='store_true')
    parser.add_argument('--path', type=str)
    parser.add_argument('--r', type=float, default=0)
    parser.add_argument('--min_sample', type=int, default=0)
    parser.add_argument('--max_size', type=int, default=0)
    parser.add_argument('--step', type=float, default=0)
    args = parser.parse_args()

    model = Net(256).to("cuda")

    if args.train:
        args.data = "casia"
        train(30, args, model)

    if args.test:
        args.data = "512"

    model.load_state_dict(torch.load("best.pth"))
    if args.cdp:
        test_gcn_cdp(model, args)
    else:
        test_gcn_dbscan(model, args)


if __name__ == '__main__':
    main()
