"""Poison Frogs replication: one command per step.

    python run.py <step> [options]       python run.py <step> --help   for that step's options

Steps, in order:
    prepare        build both datasets from the dog / not-dog archive
    transfer       part 1: clean model, craft poisons, retrain, figures 1-4
    animate        part 1: GIF of one poison being crafted pixel by pixel
    beta-sweep     part 1: strip of poisons made with different beta values
    watermark-gif  part 2: GIF of the watermark blend from 0% to 100% opacity
    e2e-pretrain   part 2: train the clean 32x32 AlexNet (cached)
    e2e-samples    part 2: grid of the 32x32 training images
    e2e-attack     part 2: one end-to-end attack with N poisons
    e2e-ablation   part 2: success rate vs. number of poisons and opacity (long)
    all            every step above, in order
"""
import argparse
import time
from pathlib import Path

from poisonfrogs import config


def step(title):
    print(f"\n=== {title} ===")
    return time.time()


def done(t0):
    print(f"(done in {time.time() - t0:.0f}s)")


# --- commands -----------------------------------------------------------------------

def cmd_prepare(a):
    from poisonfrogs import data
    archive = Path(a.archive) if a.archive else config.find_archive()
    if not archive.exists():
        raise SystemExit(f"Dataset archive not found at {archive}. Pass --archive PATH.")
    if not a.e2e_only:
        t0 = step(f"part 1 data: 224x224 subset from {archive.name}")
        data.prepare_transfer(archive)
        done(t0)
    if not a.transfer_only:
        t0 = step(f"part 2 data: whole archive at 32x32 from {archive.name}")
        data.prepare_e2e(archive)
        done(t0)


def cmd_transfer(a):
    from poisonfrogs import plots, transfer
    t0 = step("part 1: transfer learning (frozen ResNet-50, last layer retrained)")
    s = transfer.load_setup(a.target)
    r = transfer.run(s, a.n_poisons, a.beta, a.lr, a.steps)
    for path in (plots.clean_prediction(s, r), plots.poisons_grid(r), plots.targeted(s, r), plots.feature_space(s, r)):
        print("wrote", path)
    print(f"\n{'':32s}{'test acc':>10s}{'P(dog | target)':>18s}")
    print(f"{'clean':32s}{r.clean_acc:>10.1%}{r.clean_p:>18.1%}")
    for k, acc, p in r.dose:
        print(f"{f'+ {k} poison(s)':32s}{acc:>10.1%}{p:>18.1%}")
    done(t0)


def cmd_animate(a):
    from poisonfrogs import animations, transfer
    t0 = step(f"part 1: pixel-by-pixel animation (beta {a.beta:g}, lr {a.lr:g}, {a.steps} steps)")
    s = transfer.load_setup(a.target)
    out = config.FIGURES / a.out if a.out else None
    print("wrote", animations.pixel_shift(s, a.beta, a.lr, a.steps, a.amplify, a.image_only, out))
    done(t0)


def cmd_beta_sweep(a):
    from poisonfrogs import plots, transfer
    t0 = step(f"part 1: beta sweep {a.betas}")
    s = transfer.load_setup(a.target)
    print("wrote", plots.beta_sweep(s, *transfer.beta_sweep(s, a.betas, a.lr, a.steps), name=a.out))
    done(t0)


def cmd_watermark_gif(a):
    from poisonfrogs import animations
    t0 = step("part 2: watermark animation")
    print("wrote", animations.watermark_gif(Path(a.base), Path(a.target), config.FIGURES / a.out))
    done(t0)


def cmd_e2e_pretrain(a):
    from poisonfrogs import data, e2e
    t0 = step("part 2: clean 32x32 AlexNet")
    d = data.load_e2e()
    if config.E2E_WEIGHTS.exists() and not a.force:
        print(f"using cached {config.E2E_WEIGHTS.name} (pass --force to retrain)")
    model = e2e.pretrain_clean(d, force=a.force)
    target = e2e.load_target(a.target)
    print(f"test accuracy {e2e.accuracy(model, *d['test']):.1%}, "
          f"P(dog | target) {e2e.predict(model, target)[0, 1].item():.1%}")
    done(t0)


def cmd_e2e_samples(a):
    from poisonfrogs import data, plots
    t0 = step("part 2: sample of the 32x32 images")
    print("wrote", plots.e2e_samples(data.load_e2e()))
    done(t0)


def cmd_e2e_attack(a):
    from poisonfrogs import data, e2e
    t0 = step(f"part 2: one end-to-end attack ({a.n_poisons} poisons, {a.opacity:.0%} watermark)")
    d = data.load_e2e()
    model = e2e.pretrain_clean(d)
    target = e2e.load_target(a.target)
    print(f"clean model: P(dog | target) {e2e.predict(model, target)[0, 1].item():.1%}")
    e2e.attack(model, d, target, a.n_poisons, a.opacity, a.seed)
    done(t0)


def cmd_e2e_ablation(a):
    import json
    from poisonfrogs import data, e2e, plots
    out = config.DATA / a.out
    if a.plot_only:
        if not out.exists():
            raise SystemExit(f"{out} not found: run the ablation first")
        print("wrote", plots.e2e_ablation(e2e.summarize(json.loads(out.read_text())), name=Path(a.out).stem + ".png"))
        return
    runs = len(a.opacities) * a.trials * len(a.poisons)
    t0 = step(f"part 2: ablation, {runs} full retrains (about 1 minute each on a laptop GPU)")
    d = data.load_e2e()
    model = e2e.pretrain_clean(d)
    results = e2e.ablation(model, d, e2e.load_target(a.target), a.opacities, a.poisons, a.trials, out=out)
    summary = e2e.summarize(results)
    print(f"\n{'opacity':>8s}{'poisons':>9s}{'success':>9s}{'mean P(dog)':>13s}")
    for (o, n), (wins, total, mean_p) in summary.items():
        print(f"{o:>8.0%}{n:>9d}{f'{wins}/{total}':>9s}{mean_p:>13.0%}")
    print("wrote", out, "and", plots.e2e_ablation(summary, name=Path(a.out).stem + ".png"))
    done(t0)


def cmd_all(a):
    archive_ok = (config.DATA / "train").exists() and config.E2E_DATA.exists()
    if not archive_ok:
        cmd_prepare(argparse.Namespace(archive=a.archive, e2e_only=False, transfer_only=False))
    target = a.target
    cmd_transfer(argparse.Namespace(target=target, n_poisons=config.TRANSFER_N_POISONS, beta=config.TRANSFER_BETA,
                                    lr=config.TRANSFER_LR, steps=config.TRANSFER_STEPS))
    cmd_animate(argparse.Namespace(target=target, beta=config.TRANSFER_BETA, lr=config.TRANSFER_LR,
                                   steps=config.TRANSFER_STEPS, amplify=10, image_only=False, out=None))
    cmd_beta_sweep(argparse.Namespace(target=target, betas=[0, 1, 2, 5, 10, 50, 100], lr=config.TRANSFER_LR,
                                      steps=config.TRANSFER_STEPS, out="beta_sweep.png"))
    cmd_watermark_gif(argparse.Namespace(base=str(config.WATERMARK_BASE), target=str(config.WATERMARK_TARGET),
                                         out="watermark_demo.gif"))
    cmd_e2e_pretrain(argparse.Namespace(target=target, force=False))
    cmd_e2e_samples(argparse.Namespace())
    trials, poisons, out = (1, [1, 50], "e2e_ablation_quick.json") if a.quick else (5, [1, 5, 10, 25, 50], "e2e_ablation.json")
    cmd_e2e_ablation(argparse.Namespace(target=target, opacities=[0.0, 0.2, 0.3], poisons=poisons, trials=trials, out=out, plot_only=False))


# --- argument parsing ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="step")

    def add(name, fn, help_text):
        sp = sub.add_parser(name, help=help_text, description=help_text,
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        sp.set_defaults(fn=fn)
        return sp

    def add_target(sp):
        sp.add_argument("--target", default=str(config.DEFAULT_TARGET), help="image the attack should flip to 'dog'")

    sp = add("prepare", cmd_prepare, "build both datasets from the dog / not-dog archive")
    sp.add_argument("--archive", help="path to the dataset zip (default: archive.zip here or one folder up)")
    sp.add_argument("--transfer-only", action="store_true", help="only the 224x224 subset (part 1)")
    sp.add_argument("--e2e-only", action="store_true", help="only the 32x32 dataset (part 2)")

    sp = add("transfer", cmd_transfer, "part 1: clean model, poisons, retraining, figures 1-4")
    add_target(sp)
    sp.add_argument("--n-poisons", type=int, default=config.TRANSFER_N_POISONS)
    sp.add_argument("--beta", type=float, default=config.TRANSFER_BETA, help="how hard poisons are pulled back toward the dog")
    sp.add_argument("--lr", type=float, default=config.TRANSFER_LR, help="forward step size")
    sp.add_argument("--steps", type=int, default=config.TRANSFER_STEPS)

    sp = add("animate", cmd_animate, "part 1: GIF of one poison being crafted")
    add_target(sp)
    sp.add_argument("--beta", type=float, default=config.TRANSFER_BETA, help="0 = no pull back: the dog turns to noise")
    sp.add_argument("--lr", type=float, default=config.TRANSFER_LR)
    sp.add_argument("--steps", type=int, default=config.TRANSFER_STEPS)
    sp.add_argument("--amplify", type=float, default=10, help="how much the change panel is amplified (1 = not at all)")
    sp.add_argument("--image-only", action="store_true", help="animate just the dog photo")
    sp.add_argument("--out", help="file name in figures/ (default pixel_shift.gif or pixel_shift_image.gif)")

    sp = add("beta-sweep", cmd_beta_sweep, "part 1: one poison per beta value, side by side")
    add_target(sp)
    sp.add_argument("--betas", type=float, nargs="+", default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    sp.add_argument("--lr", type=float, default=config.TRANSFER_LR)
    sp.add_argument("--steps", type=int, default=config.TRANSFER_STEPS)
    sp.add_argument("--out", default="beta_sweep.png", help="file name in figures/")

    sp = add("watermark-gif", cmd_watermark_gif, "part 2: GIF of the watermark blend")
    sp.add_argument("--base", default=str(config.WATERMARK_BASE), help="base image (a dog)")
    sp.add_argument("--target", default=str(config.WATERMARK_TARGET), help="image blended in (default: a parrot)")
    sp.add_argument("--out", default="watermark_demo.gif", help="file name in figures/")

    sp = add("e2e-pretrain", cmd_e2e_pretrain, "part 2: train the clean 32x32 AlexNet (about 5 minutes, cached)")
    add_target(sp)
    sp.add_argument("--force", action="store_true", help="retrain even if cached weights exist")

    add("e2e-samples", cmd_e2e_samples, "part 2: grid of 32x32 training images")

    sp = add("e2e-attack", cmd_e2e_attack, "part 2: one end-to-end attack")
    add_target(sp)
    sp.add_argument("--n-poisons", type=int, default=50)
    sp.add_argument("--opacity", type=float, default=config.E2E_OPACITY, help="watermark opacity, 0 to 1")
    sp.add_argument("--seed", type=int, default=0, help="which random base dogs to use")

    sp = add("e2e-ablation", cmd_e2e_ablation, "part 2: success rate vs. poisons and opacity")
    add_target(sp)
    sp.add_argument("--opacities", type=float, nargs="+", default=[0.0, 0.2, 0.3])
    sp.add_argument("--poisons", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    sp.add_argument("--trials", type=int, default=5)
    sp.add_argument("--out", default="e2e_ablation.json", help="results file in data/ (the chart goes to figures/ with the same name)")
    sp.add_argument("--plot-only", action="store_true", help="redraw the chart from an existing results file, no training")

    sp = add("all", cmd_all, "every step in order (about 1 hour; --quick about 15 minutes on a first run)")
    add_target(sp)
    sp.add_argument("--archive", help="path to the dataset zip")
    sp.add_argument("--quick", action="store_true", help="ablation with 1 trial and 1 or 50 poisons only")

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
