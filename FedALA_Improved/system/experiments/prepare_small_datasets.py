"""Download four compact vision datasets and export FedALA's NPZ layout.

The generated partitions use the same 20-client Dirichlet(alpha=0.1) protocol
as the paper.  Run from the repository root or pass --out explicitly.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

DATASETS = {
    "fashion-mnist": (datasets.FashionMNIST, 1),
    "kmnist": (datasets.KMNIST, 1),
    "usps": (datasets.USPS, 1),
    "cifar10": (datasets.CIFAR10, 3),
}

def partition(xs, ys, clients, alpha, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(clients)]
    for label in np.unique(ys):
        ids = np.flatnonzero(ys == label)
        rng.shuffle(ids)
        probs = rng.dirichlet(np.full(clients, alpha))
        cuts = (np.cumsum(probs) * len(ids)).astype(int)[:-1]
        for c, part in enumerate(np.split(ids, cuts)):
            buckets[c].extend(part.tolist())
    return buckets

def export(name, cls, channels, root, out, clients, alpha):
    tfm = transforms.Compose(([transforms.Resize((28, 28))] if channels == 1 else []) + [transforms.ToTensor()])
    kwargs = dict(root=str(root), download=True, transform=tfm)
    train, test = cls(train=True, **kwargs), cls(train=False, **kwargs)
    xtr = torch.stack([x for x, _ in train]).numpy().astype('float32')
    ytr = np.asarray([y for _, y in train], dtype='int64')
    xte = torch.stack([x for x, _ in test]).numpy().astype('float32')
    yte = np.asarray([y for _, y in test], dtype='int64')
    if channels == 3:
        xtr, xte = xtr, xte
    target = Path(out) / name
    for split, xs, ys in [('train', xtr, ytr), ('test', xte, yte)]:
        (target / split).mkdir(parents=True, exist_ok=True)
        for i, ids in enumerate(partition(xs, ys, clients, alpha, 100)):
            np.savez(target / split / f'{i}.npz', data={'x': xs[ids], 'y': ys[ids]})
    (target / 'config.json').write_text(json.dumps({'num_clients': clients, 'num_classes': 10, 'alpha': alpha}))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../dataset_cache')
    ap.add_argument('--out', default='../dataset')
    ap.add_argument('--clients', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.1)
    args = ap.parse_args()
    for name, (cls, channels) in DATASETS.items():
        export(name, cls, channels, args.root, args.out, args.clients, args.alpha)
