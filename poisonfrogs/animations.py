"""GIFs: one poison being crafted pixel by pixel, and the watermark blend."""
import matplotlib
matplotlib.use("Agg")  # headless; also avoids retina backends doubling the GIF size
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches

from . import config
from .attack import craft
from .features import p_dog, train_head
from .gif import save_gif
from .images import load_image, max_change, quantize, to_numpy
from .plots import feature_map
from .transfer import load_bases, nearest_dogs

ZOOM = 8  # the zoomed window is ZOOM x ZOOM pixels


def _record(s, beta, lr, steps):
    """Craft one poison from the dog nearest the target, keeping snapshots on a log scale of steps
    (most of the movement happens early)."""
    base = load_bases(s, nearest_dogs(s, 1)[0])
    keep = set(np.unique(np.round(np.geomspace(1, steps, 90))).astype(int)) | {0}
    snaps, feats, kept = [], [], []

    def on_step(step, x):
        if step in keep:
            xq = quantize(x)
            with torch.no_grad():
                feats.append(s.fx(xq)[0].cpu())
            snaps.append(xq[0].cpu())
            kept.append(step)

    craft(s.fx, base, s.target[None], beta, lr, steps, on_step=on_step)
    dists = [(f - s.f_target).norm().item() for f in feats]
    return base[0], snaps, feats, kept, dists


def pixel_shift(s, beta=config.TRANSFER_BETA, lr=config.TRANSFER_LR, steps=config.TRANSFER_STEPS,
                amplify=10, image_only=False, out=None, log=print):
    """Full view: the dog being edited, an 8x8 zoom before/now, the change, the feature distance,
    and the poison walking onto the target in a 2-D map of feature space. image_only: just the dog."""
    out = out or config.FIGURES / ("pixel_shift_image.gif" if image_only else "pixel_shift.gif")
    base, snaps, feats, kept, dists = _record(s, beta, lr, steps)
    log(f"feature distance {dists[0]:.1f} -> {dists[-1]:.1f}")

    if image_only:
        fig, ax = plt.subplots(figsize=(5, 5.5), dpi=100)
        im = ax.imshow(to_numpy(base), interpolation="nearest")
        ax.axis("off")
        caption = ax.set_title("", fontsize=12, fontweight="bold")
        order = [0] * 12 + list(range(len(snaps))) + [len(snaps) - 1] * 36

        def draw_image(k):
            i = order[k]
            im.set_data(to_numpy(snaps[i]))
            caption.set_text(f"β = {beta:g}   |   step {kept[i]}\nbiggest pixel change: {max_change(snaps[i], base)}/255")

        save_gif(fig, draw_image, len(order), out)
        plt.close(fig)
        return out

    # what retraining with this single poison does
    clean_p = p_dog(train_head(s.X_train, s.y_train), s.f_target[None]).item()
    head = train_head(torch.cat([s.X_train, feats[-1][None]]), torch.cat([s.y_train, torch.tensor([1])]))
    poison_p = p_dog(head, s.f_target[None]).item()
    log(f"P(dog | target) after retraining with this one poison: {clean_p:.1%} -> {poison_p:.1%}")

    # zoom window: where the final change is largest
    delta = (snaps[-1] - base).abs().sum(0)
    win = F.avg_pool2d(delta[None, None], ZOOM, stride=2)[0, 0]
    r, c = divmod(win.argmax().item(), win.shape[1])
    r0, c0 = r * 2, c * 2
    crop = lambda img: to_numpy(img[:, r0:r0 + ZOOM, c0:c0 + ZOOM])

    proj = feature_map(s.X_train)
    P_tr, P_t, P_path = proj(s.X_train), proj(s.f_target[None])[0], proj(torch.stack(feats))

    fig = plt.figure(figsize=(15, 7.2), dpi=90)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.35], hspace=0.28, wspace=0.18)
    ax_full, ax_tgt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_orig, ax_cur = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_diff, ax_dist = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    ax_map = fig.add_subplot(gs[:, 3])

    def pixel_grid(ax):
        ax.set_xticks(np.arange(-0.5, ZOOM, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ZOOM, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.6)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)

    im_full = ax_full.imshow(to_numpy(base))
    ax_full.add_patch(patches.Rectangle((c0 - 0.5, r0 - 0.5), ZOOM, ZOOM, fill=False, ec="red", lw=2))
    ax_full.set_title("Dog photo being edited", fontsize=11)
    ax_full.axis("off")
    ax_tgt.imshow(to_numpy(s.target))
    ax_tgt.set_title("Target: my photo", fontsize=11)
    ax_tgt.axis("off")
    ax_orig.imshow(crop(base), interpolation="nearest")
    ax_orig.set_title(f"Original pixels ({ZOOM}x{ZOOM} zoom)", fontsize=11)
    pixel_grid(ax_orig)
    im_cur = ax_cur.imshow(crop(base), interpolation="nearest")
    ax_cur.set_title("Poison pixels now", fontsize=11)
    pixel_grid(ax_cur)
    im_diff = ax_diff.imshow(np.full((ZOOM, ZOOM, 3), 0.5), interpolation="nearest")
    ax_diff.set_title(("Change, not amplified" if amplify == 1 else f"Change, amplified x{amplify:g}")
                      + "\n(numbers = largest channel shift, out of 255)", fontsize=10)
    pixel_grid(ax_diff)
    cell_text = [[ax_diff.text(j, i, "", ha="center", va="center", fontsize=8, fontweight="bold")
                  for j in range(ZOOM)] for i in range(ZOOM)]

    ax_dist.plot(kept, dists, color="lightgray", lw=2)
    dist_line, = ax_dist.plot([], [], color="crimson", lw=2)
    dist_dot, = ax_dist.plot([], [], "o", color="crimson")
    ax_dist.set_xscale("symlog", linthresh=10)
    ax_dist.set_xlim(0, steps)
    ax_dist.set_ylim(0, dists[0] * 1.1)
    ax_dist.set_xlabel("optimization step")
    ax_dist.set_title("Feature distance to my photo", fontsize=11)
    ax_dist.grid(alpha=0.3)

    ax_map.scatter(*P_tr[s.y_train == 1].T, s=5, alpha=0.25, c="tab:blue", label="dog photos")
    ax_map.scatter(*P_tr[s.y_train == 0].T, s=5, alpha=0.25, c="tab:orange", label="not-dog photos")
    ax_map.scatter(*P_t, s=320, marker="*", c="crimson", edgecolors="black", zorder=5, label="my photo")
    ax_map.scatter(*P_path[0], s=70, c="tab:blue", edgecolors="black", zorder=4, label="original dog")
    trail, = ax_map.plot([], [], "-", color="black", lw=1.2, zorder=4)
    head_dot = ax_map.scatter([], [], s=110, marker="X", c="black", zorder=6, label="poison")
    ax_map.legend(loc="lower left", fontsize=9)
    ax_map.set_title("What the network sees (2-D projection)", fontsize=11)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    suptitle = fig.suptitle("", fontsize=15, fontweight="bold")
    banner = fig.text(0.5, 0.015, "", ha="center", fontsize=14, color="crimson", fontweight="bold")
    hold_start, hold_end = 15, 45
    order = [0] * hold_start + list(range(len(snaps))) + [len(snaps) - 1] * hold_end

    def draw(k):
        i = order[k]
        d = snaps[i] - base
        im_full.set_data(to_numpy(snaps[i]))
        im_cur.set_data(crop(snaps[i]))
        im_diff.set_data(np.clip(0.5 + amplify * d[:, r0:r0 + ZOOM, c0:c0 + ZOOM].permute(1, 2, 0).numpy(), 0, 1))
        dc = d[:, r0:r0 + ZOOM, c0:c0 + ZOOM]
        signed = (dc.gather(0, dc.abs().argmax(0, keepdim=True))[0] * 255).round().int().numpy()
        for a in range(ZOOM):
            for b in range(ZOOM):
                cell_text[a][b].set_text(f"{signed[a, b]:+d}" if signed[a, b] else "0")
        dist_line.set_data(kept[:i + 1], dists[:i + 1])
        dist_dot.set_data([kept[i]], [dists[i]])
        trail.set_data(P_path[:i + 1, 0], P_path[:i + 1, 1])
        head_dot.set_offsets(P_path[i:i + 1])
        suptitle.set_text(f"β = {beta:g}   |   Step {kept[i]:4d}   |   biggest pixel change: "
                          f"{max_change(snaps[i], base)}/255   |   feature distance: {dists[i]:.1f}")
        banner.set_text(f"Retrain with this ONE image labeled \"dog\"  ->  P(dog | my photo): {clean_p:.0%} -> {poison_p:.0%}"
                        if k >= hold_start + len(snaps) else "")

    save_gif(fig, draw, len(order), out)
    plt.close(fig)
    return out


def watermark_gif(base_path=config.WATERMARK_BASE, target_path=config.WATERMARK_TARGET,
                  out=config.FIGURES / "watermark_demo.gif", paper_opacities=(20, 30)):
    """The paper's watermark blend, b <- gamma*t + (1-gamma)*b, as gamma goes 0% -> 100%."""
    for p in (base_path, target_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run: python run.py prepare (or pass --base / --target)")
    base = to_numpy(load_image(base_path, config.TRANSFER_SIZE))
    target = to_numpy(load_image(target_path, config.TRANSFER_SIZE))
    hold = {0: 12, 20: 12, 30: 18, 100: 18}
    order = [g for g in range(0, 101, 2) for _ in range(hold.get(g, 1))]

    fig = plt.figure(figsize=(10, 4.4), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.07], hspace=0.3, wspace=0.08)
    ax_b, ax_w, ax_t = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_bar = fig.add_subplot(gs[1, :])
    for ax, img, title in ((ax_b, base, "Base b (dog)"), (ax_t, target, "Target t")):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    im_w = ax_w.imshow(base)
    ax_w.axis("off")
    title_w = ax_w.set_title("", fontsize=12, fontweight="bold")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 20, 30, 50, 100], ["0%", "20%", "30%", "50%", "100%"])
    ax_bar.set_xlabel("opacity γ of the target (paper's settings in red)", fontsize=10)
    for g in paper_opacities:
        ax_bar.axvline(g, color="crimson", lw=2, zorder=3)
    fill = ax_bar.barh(0.5, 0, height=1, color="0.35")[0]
    fig.suptitle("Watermarking:  b  ←  γ · t  +  (1 − γ) · b", fontsize=15, fontweight="bold")
    note = ax_bar.set_title("", fontsize=11, color="crimson", fontweight="bold")

    def draw(k):
        g = order[k]
        im_w.set_data(np.clip(g / 100 * target + (1 - g / 100) * base, 0, 1))
        title_w.set_text(f"Watermarked base, γ = {g}%")
        fill.set_width(g)
        note.set_text({30: "the paper's main setting (30%)", 20: "also tested in the paper (20%)"}.get(g, ""))

    save_gif(fig, draw, len(order), out)
    plt.close(fig)
    return out
