"""Static figures. Every function saves a PNG in figures/ and returns its path."""
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .images import load_image, to_numpy


def _save(fig, name):
    config.FIGURES.mkdir(exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_map(X_train):
    """2-D PCA of the training features: a rough picture of 'what the network sees'."""
    mu = X_train.mean(0)
    _, _, V = torch.pca_lowrank(X_train - mu, q=2, center=False)
    return lambda Z: ((Z - mu) @ V).numpy()


# --- part 1 -------------------------------------------------------------------------

def clean_prediction(s, r):
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(to_numpy(s.target))
    plt.title(f"Clean model: P(dog) = {r.clean_p:.1%}")
    plt.axis("off")
    return _save(fig, "1_clean_prediction.png")


def poisons_grid(r, amplify=10):
    n = len(r.poisons)
    fig, ax = plt.subplots(3, n, figsize=(2.4 * n, 7.5), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(to_numpy(r.bases[i]))
        ax[1, i].imshow(to_numpy(r.poisons[i]))
        ax[2, i].imshow(to_numpy(0.5 + amplify * (r.poisons[i] - r.bases[i])))
        for row in range(3):
            ax[row, i].axis("off")
    ax[0, 0].set_title("base (clean dog)", loc="left")
    ax[1, 0].set_title("poison (labeled 'dog')", loc="left")
    ax[2, 0].set_title(f"perturbation x{amplify}", loc="left")
    fig.tight_layout()
    return _save(fig, "2_poisons.png")


def targeted(s, r):
    """The target flips; other not-dog images (faces) don't."""
    poisoned_p = r.dose[-1][2]
    n = len(r.compare_paths) + 1
    fig, ax = plt.subplots(1, n, figsize=(2.4 * n, 3))
    ax[0].imshow(to_numpy(s.target))
    ax[0].set_title(f"target\n{r.clean_p:.0%} -> {poisoned_p:.0%} dog", color="crimson")
    for a, p, b, af in zip(ax[1:], r.compare_paths, r.compare_before, r.compare_after):
        a.imshow(to_numpy(load_image(p, config.TRANSFER_SIZE)))
        a.set_title(f"other face\n{b:.0%} -> {af:.0%} dog")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return _save(fig, "3_targeted.png")


def feature_space(s, r):
    proj = feature_map(s.X_train)
    P_tr, P_b, P_p, P_t = proj(s.X_train), proj(s.X_train[r.base_idx]), proj(r.X_poison), proj(s.f_target[None])
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(*P_tr[s.y_train == 1].T, s=6, alpha=0.25, c="tab:blue", label="dog (train)")
    plt.scatter(*P_tr[s.y_train == 0].T, s=6, alpha=0.25, c="tab:orange", label="not dog (train)")
    for b, p in zip(P_b, P_p):
        plt.annotate("", xy=p, xytext=b, arrowprops=dict(arrowstyle="->", color="black", lw=1))
    plt.scatter(*P_b.T, s=50, c="tab:blue", edgecolors="black", label="base dogs")
    plt.scatter(*P_p.T, s=60, marker="X", c="black", label="poisons")
    plt.scatter(*P_t.T, s=250, marker="*", c="crimson", edgecolors="black", label="target photo")
    plt.legend(loc="best", fontsize=8)
    plt.title("Feature collision: poisons land on the target")
    plt.xticks([])
    plt.yticks([])
    return _save(fig, "4_feature_space.png")


def beta_sweep(s, base, base_dist, poisons, rows, name="beta_sweep.png", amplify=10):
    """One column per beta: the poison on top, its change (amplified) underneath."""
    n = len(rows) + 2
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.4),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.06})
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(to_numpy(s.target))
    axes[0, 0].set_title("Target: my photo", fontsize=10, fontweight="bold", color="crimson")
    axes[0, 1].imshow(to_numpy(base))
    axes[0, 1].set_title(f"Original dog\ndistance {base_dist:.1f}", fontsize=10, fontweight="bold")
    axes[1, 1].text(0.5, 0.5, f"change x{amplify}\n(grey = no change)", ha="center", va="center",
                    fontsize=10, transform=axes[1, 1].transAxes)
    for k, r in enumerate(rows):
        col = k + 2
        axes[0, col].imshow(to_numpy(poisons[k]))
        axes[0, col].set_title(f"β = {r['beta']:g}\nPSNR {r['psnr']:.0f} dB, max {r['max']}/255", fontsize=9)
        axes[1, col].imshow(to_numpy(0.5 + amplify * (poisons[k] - base)))
        flipped = r["p_dog"] > 0.5
        axes[1, col].set_title(f"distance {r['dist']:.1f}\nP(dog | me) {r['p_dog']:.0%}", fontsize=9,
                               color="crimson" if flipped else "black", fontweight="bold" if flipped else "normal")
    fig.suptitle("Same dog, one poison per β: the smaller β is, the further the pixels drift from the original dog",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, name)


# --- part 2 -------------------------------------------------------------------------

def e2e_samples(data, n=8, seed=1):
    x, y = data["train"]
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.where(y.numpy() == c)[0], n, replace=False) for c in (1, 0)]
    fig, ax = plt.subplots(2, n, figsize=(12, 3.6))
    for row, (idx, name) in enumerate(zip(picks, ("dog", "not dog"))):
        for col, i in enumerate(idx):
            ax[row, col].imshow(to_numpy(x[i]), interpolation="nearest")
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
        ax[row, 0].set_ylabel(name, fontsize=12)
    fig.suptitle("32 x 32 training images used in the end-to-end experiments (shown enlarged, no smoothing)", fontsize=12)
    fig.tight_layout()
    return _save(fig, "e2e_32_samples.png")


def e2e_ablation(summary, name="e2e_ablation.png"):
    """Grouped bars: success rate per number of poisons, one bar per watermark opacity."""
    opacities = sorted({o for o, _ in summary})
    ns = sorted({n for _, n in summary})
    colors = {0.0: "#B9C9BF", 0.2: "#276B43", 0.3: "#B03A0B"}
    width = 0.8 / len(opacities)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, o in enumerate(opacities):
        rates = [100 * summary[(o, n)][0] / summary[(o, n)][1] if (o, n) in summary else 0 for n in ns]
        ax.bar(np.arange(len(ns)) + (j - (len(opacities) - 1) / 2) * width, rates, width,
               label="no watermark" if o == 0 else f"{o:.0%} watermark", color=colors.get(o))
    trials = max(t for _, t, _ in summary.values())
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("number of poisons")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"End-to-end attack success ({trials} trials per bar)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)
