"""Part 1: the one-shot kill on transfer learning (paper Sec. 3).

The victim keeps ResNet-50 frozen and retrains only the last layer. The attacker
crafts a few poisons from the dogs nearest the target in feature space, the victim
adds them (labeled "dog") and retrains, and we check what the model says about the target.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from . import config
from .attack import craft
from .data import transfer_split
from .features import ResNetFeatures, accuracy, cached_features, p_dog, train_head
from .images import load_image, max_change, psnr, save_png


@dataclass
class Setup:
    """Everything the part 1 steps share: the extractor, the data and the target."""
    fx: ResNetFeatures
    train_paths: list
    y_train: torch.Tensor
    test_paths: list
    y_test: torch.Tensor
    X_train: torch.Tensor
    X_test: torch.Tensor
    target: torch.Tensor        # (3, 224, 224)
    f_target: torch.Tensor      # (2048,)


def load_setup(target_path=config.DEFAULT_TARGET):
    target_path = Path(target_path)
    if not target_path.exists():
        raise SystemExit(f"Target image {target_path} not found. Put a photo there or pass --target.")
    fx = ResNetFeatures()
    train_paths, y_train = transfer_split("train")
    test_paths, y_test = transfer_split("test")
    X_train, X_test = cached_features(fx, train_paths, test_paths)
    target = load_image(target_path, config.TRANSFER_SIZE)
    with torch.no_grad():
        f_target = fx(target[None])[0].cpu()
    return Setup(fx, train_paths, y_train, test_paths, y_test, X_train, X_test, target, f_target)


def nearest_dogs(s, n):
    """Indices of the n training dogs whose features are already closest to the target, and their distances."""
    dog_idx = (s.y_train == 1).nonzero().squeeze(1)
    dist = (s.X_train[dog_idx] - s.f_target).norm(dim=1)
    order = dist.argsort()[:n]
    return dog_idx[order], dist[order]


def load_bases(s, idx):
    return torch.stack([load_image(s.train_paths[i], config.TRANSFER_SIZE) for i in idx])


@dataclass
class Result:
    clean_head: torch.nn.Module
    clean_acc: float
    clean_p: float                  # P(dog | target) before the attack
    base_idx: torch.Tensor
    base_dists: torch.Tensor
    typical_dog_distance: float     # for scale: median distance between two dog photos
    bases: torch.Tensor
    poisons: torch.Tensor
    X_poison: torch.Tensor          # poison features, read back from the saved PNGs
    dose: list                      # [(n poisons, test accuracy, P(dog | target)), ...]
    poisoned_head: torch.nn.Module
    compare_paths: list             # other not-dog test images, to show the attack is targeted
    compare_before: torch.Tensor
    compare_after: torch.Tensor


def run(s, n_poisons=config.TRANSFER_N_POISONS, beta=config.TRANSFER_BETA,
        lr=config.TRANSFER_LR, steps=config.TRANSFER_STEPS, log=print):
    clean_head = train_head(s.X_train, s.y_train)
    clean_acc = accuracy(clean_head, s.X_test, s.y_test)
    clean_p = p_dog(clean_head, s.f_target[None]).item()
    log(f"clean model: test accuracy {clean_acc:.1%}, P(dog | target) {clean_p:.1%}")

    base_idx, base_dists = nearest_dogs(s, n_poisons)
    dog_feats = s.X_train[s.y_train == 1][:200]
    pairwise = torch.cdist(dog_feats, dog_feats)
    typical = pairwise[pairwise > 0].median().item()
    log(f"base dogs: feature distance to target {[round(d, 1) for d in base_dists.tolist()]} "
        f"(two random dog photos are about {typical:.0f} apart)")

    bases = load_bases(s, base_idx)

    def progress(step, x):
        if step % 200 == 0:
            with torch.no_grad():
                d = (s.fx(x).cpu() - s.f_target).norm(dim=1).mean().item()
            log(f"  step {step:4d}: mean feature distance {d:.2f}")

    poisons = craft(s.fx, bases, s.target[None], beta, lr, steps, on_step=progress)
    for old in config.POISONS.glob("poison_*.png"):
        old.unlink()
    poison_paths = [config.POISONS / f"poison_{i}.png" for i in range(n_poisons)]
    for img, path in zip(poisons, poison_paths):
        save_png(img, path)
    for i in range(n_poisons):
        log(f"poison {i}: PSNR {psnr(poisons[i], bases[i]):.1f} dB, "
            f"max pixel change {max_change(poisons[i], bases[i])}/255")

    # the victim reads the poisons from disk, labels them "dog" and retrains the last layer
    X_poison = s.fx.extract(poison_paths)
    log(f"poison feature distance to target: {[round(d, 1) for d in (X_poison - s.f_target).norm(dim=1).tolist()]}")
    dose = []
    for k in range(1, n_poisons + 1):
        head = train_head(torch.cat([s.X_train, X_poison[:k]]),
                          torch.cat([s.y_train, torch.ones(k, dtype=torch.long)]))
        dose.append((k, accuracy(head, s.X_test, s.y_test), p_dog(head, s.f_target[None]).item()))
        log(f"{k} poison(s): test accuracy {dose[-1][1]:.1%}, P(dog | target) {dose[-1][2]:.1%}")

    compare_paths = [p for i in config.COMPARE_INDICES
                     if (p := config.DATA / "test" / "other" / f"other_{i:04d}.png").exists()]
    X_cmp = s.fx.extract(compare_paths)
    return Result(clean_head, clean_acc, clean_p, base_idx, base_dists, typical, bases, poisons,
                  X_poison, dose, head, compare_paths, p_dog(clean_head, X_cmp), p_dog(head, X_cmp))


def beta_sweep(s, betas, lr=config.TRANSFER_LR, steps=config.TRANSFER_STEPS, log=print):
    """One poison per beta, all from the same base dog (the one nearest the target).

    For each: how visible the change is, how close the features got, and what the
    victim says about the target after retraining with that single poison.
    """
    base_idx, base_dist = nearest_dogs(s, 1)
    base = load_bases(s, base_idx)[0]
    poisons = craft(s.fx, base.expand(len(betas), -1, -1, -1), s.target[None], betas, lr, steps)
    with torch.no_grad():
        f_poison = s.fx(poisons).cpu()
    rows = []
    for k, beta in enumerate(betas):
        head = train_head(torch.cat([s.X_train, f_poison[k:k + 1]]), torch.cat([s.y_train, torch.tensor([1])]))
        rows.append({"beta": beta, "psnr": psnr(poisons[k], base), "max": max_change(poisons[k], base),
                     "dist": (f_poison[k] - s.f_target).norm().item(),
                     "p_dog": p_dog(head, s.f_target[None]).item()})
        r = rows[-1]
        log(f"beta {beta:6g}: PSNR {r['psnr']:5.1f} dB, max change {r['max']:3d}/255, "
            f"distance {r['dist']:4.1f}, P(dog | target) {r['p_dog']:.1%}")
    return base, base_dist.item(), poisons, rows
