"""Part 2: end-to-end training (paper Sec. 4).

The victim retrains EVERY layer, so the feature extractor can move and push a
single poison back among the dogs. The paper's fix: many diverse poisons, each
with a faint watermark of the target. Everything here is 32x32 with the paper's
scaled-down AlexNet (Appendix F, Table 1).
"""
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .attack import craft, watermark
from .images import load_image


class SmallAlexNet(nn.Module):
    """Paper Table 1: (conv 5x5, 64 + ReLU, max-pool 3x3/2, LRN) x 2, then FC 384 -> 192 -> classes."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1), nn.LocalResponseNorm(4, alpha=0.001 / 9, beta=0.75, k=1),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1), nn.LocalResponseNorm(4, alpha=0.001 / 9, beta=0.75, k=1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(),
        )
        self.out = nn.Linear(192, n_classes)

    def features(self, x):
        # x in [0, 1]; the 192-d penultimate layer is the "feature space" f
        return self.fc(self.conv(x - 0.5))

    def forward(self, x):
        return self.out(self.features(x))


def augment(x, g):
    """Random horizontal flip + random 4-pixel shift (standard CIFAR augmentation)."""
    flip = torch.rand(len(x), generator=g) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,), generator=g).tolist()
    return pad[:, :, i:i + x.shape[2], j:j + x.shape[3]]


def train(model, x, y, epochs, lr, final_lr=None, seed=0, log=None, aug=False):
    """Adam on the full network. With final_lr, the learning rate decays (cosine) from lr to final_lr."""
    g = torch.Generator().manual_seed(seed)
    model.train().to(config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * ((len(x) + config.E2E_BATCH - 1) // config.E2E_BATCH)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=final_lr)
             if final_lr is not None else None)
    for ep in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), config.E2E_BATCH):
            idx = perm[i:i + config.E2E_BATCH]
            xb = augment(x[idx], g) if aug else x[idx]
            loss = F.cross_entropy(model(xb.to(config.DEVICE)), y[idx].to(config.DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()
        if log:
            log(ep, model)
    return model.eval()


@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i + bs].to(config.DEVICE)).softmax(1).cpu() for i in range(0, len(x), bs)])


def accuracy(model, x, y):
    return (predict(model, x).argmax(1) == y).float().mean().item()


def load_target(path=config.DEFAULT_TARGET):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Target image {path} not found. Put a photo there or pass --target.")
    return load_image(path, config.E2E_SIZE)[None]


def pretrain_clean(data, force=False, log=print):
    """The victim's clean model (the paper's warm start). Cached after the first run (~5 min)."""
    model = SmallAlexNet()
    if config.E2E_WEIGHTS.exists() and not force:
        model.load_state_dict(torch.load(config.E2E_WEIGHTS, map_location="cpu"))
        return model.to(config.DEVICE).eval()
    xtr, ytr = data["train"]
    xte, yte = data["test"]

    def progress(ep, m):
        if ep % 10 == 9 or ep == config.E2E_PRETRAIN_EPOCHS - 1:
            log(f"  epoch {ep + 1:3d}: train accuracy {accuracy(m, xtr, ytr):.1%}, test accuracy {accuracy(m, xte, yte):.1%}")
            m.train()

    torch.manual_seed(0)
    # augmentation keeps the small network from memorising; without it a bearded face scored 99% dog
    train(model, xtr, ytr, config.E2E_PRETRAIN_EPOCHS, lr=1e-3, final_lr=config.E2E_RETRAIN_LR,
          log=progress, aug=True)
    torch.save(model.state_dict(), config.E2E_WEIGHTS)
    return model


def retrain_with_poisons(clean_model, data, poisons, seed=0):
    """The victim retrains ALL layers on the clean training data + the poisons (labeled dog)."""
    xtr, ytr = data["train"]
    x = torch.cat([xtr, poisons])
    y = torch.cat([ytr, torch.ones(len(poisons), dtype=torch.long)])
    model = copy.deepcopy(clean_model)
    torch.manual_seed(seed)
    return train(model, x, y, config.E2E_RETRAIN_EPOCHS, lr=config.E2E_RETRAIN_LR, seed=seed)


def make_poisons(model, data, target, n, opacity, seed, beta=config.E2E_BETA,
                 lr=config.E2E_LR, steps=config.E2E_STEPS):
    """n poisons from n random dogs in the attacker's pool, each watermarked with the target first."""
    xp, yp = data["pool"]
    pool_dogs = xp[yp == 1]
    idx = torch.randperm(len(pool_dogs), generator=torch.Generator().manual_seed(seed))[:n]
    return craft(model.features, watermark(pool_dogs[idx], target, opacity), target, beta, lr, steps)


def attack(model, data, target, n=50, opacity=config.E2E_OPACITY, seed=0, log=print):
    """One end-to-end attack: craft n poisons, retrain everything, report P(dog | target)."""
    poisons = make_poisons(model, data, target, n, opacity, seed)
    poisoned = retrain_with_poisons(model, data, poisons, seed=seed)
    p, acc = predict(poisoned, target)[0, 1].item(), accuracy(poisoned, *data["test"])
    log(f"{n} poisons, {opacity:.0%} watermark: P(dog | target) {p:.1%}, test accuracy {acc:.1%}"
        f" -> {'success' if p > 0.5 else 'failed'}")
    return poisons, p, acc


def ablation(model, data, target, opacities, n_poisons, trials, out=config.E2E_ABLATION, log=print):
    """Success rate vs. number of poisons and watermark opacity (the paper's Fig. 5b).

    Each trial draws fresh random base dogs and crafts max(n_poisons) poisons once;
    smaller counts use the first n of them. Results are saved after every retrain.
    """
    results = []
    t0 = time.time()
    for opacity in opacities:
        for trial in range(trials):
            poisons = make_poisons(model, data, target, max(n_poisons), opacity, seed=trial)
            for n in n_poisons:
                m = retrain_with_poisons(model, data, poisons[:n], seed=trial)
                p = predict(m, target)[0, 1].item()
                results.append({"opacity": opacity, "trial": trial, "n_poisons": n, "p_dog": p,
                                "test_acc": accuracy(m, *data["test"])})
                log(f"  opacity {opacity:.0%}, trial {trial}, {n:2d} poisons: P(dog | target) {p:.3f} "
                    f"({time.time() - t0:.0f}s)")
                out.write_text(json.dumps(results, indent=1))
    return results


def summarize(results):
    """{(opacity, n): (successes, trials, mean P(dog))} from ablation results."""
    cells = {}
    for r in results:
        cells.setdefault((r["opacity"], r["n_poisons"]), []).append(r["p_dog"])
    return {k: (sum(p > 0.5 for p in v), len(v), sum(v) / len(v)) for k, v in sorted(cells.items())}
