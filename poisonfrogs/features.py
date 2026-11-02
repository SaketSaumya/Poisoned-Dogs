"""Part 1's victim model: a frozen ImageNet ResNet-50 with one trainable linear layer on top.

f(x) = the 2,048 numbers ResNet-50 produces just before its classification layer.
The last layer (2,048 -> 2, dog / not dog) is the only part the victim trains.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import config
from .images import load_image


class ResNetFeatures:
    """Frozen feature extractor: images in [0, 1] -> 2,048-d feature vectors."""

    def __init__(self):
        net = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        net.fc = nn.Identity()
        self.net = net.eval().to(config.DEVICE)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=config.DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=config.DEVICE).view(1, 3, 1, 1)

    def __call__(self, x):
        """Differentiable: Algorithm 1 backpropagates through this to the pixels."""
        return self.net((x.to(config.DEVICE) - self.mean) / self.std)

    @torch.no_grad()
    def extract(self, paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            batch = torch.stack([load_image(p, config.TRANSFER_SIZE) for p in paths[i:i + bs]])
            out.append(self(batch).cpu())
        return torch.cat(out)


def cached_features(fx, train_paths, test_paths):
    """Features of every train/test image, computed once and saved (about 15 s on a laptop GPU)."""
    if config.FEATURES_CACHE.exists():
        return torch.load(config.FEATURES_CACHE)
    t0 = time.time()
    X_train, X_test = fx.extract(train_paths), fx.extract(test_paths)
    torch.save((X_train, X_test), config.FEATURES_CACHE)
    print(f"  extracted features in {time.time() - t0:.0f}s")
    return X_train, X_test


def train_head(X, y, epochs=config.HEAD_EPOCHS):
    """The victim's last layer: full-batch logistic regression, trained to convergence like the paper."""
    torch.manual_seed(0)
    head = nn.Linear(X.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(X), y).backward()
        opt.step()
    return head


@torch.no_grad()
def p_dog(head, feats):
    return head(feats).softmax(1)[:, 1]


@torch.no_grad()
def accuracy(head, X, y):
    return (head(X).argmax(1) == y).float().mean().item()
