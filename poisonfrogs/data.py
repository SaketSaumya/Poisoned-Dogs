"""Building the two datasets from the dog / not-dog archive.

The archive is a zip with "<top folder>/dog/*.jpg" and "<top folder>/other/*.jpg"
(about 12.5k images each). Both builders shuffle with a fixed seed, so the same
archive always gives the same splits.
"""
import io
import random
import sys
import zipfile

import numpy as np
import torch
from PIL import Image

from . import config


def _class_files(z):
    names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    return {cls: sorted(n for n in names if n.split("/")[-2] == cls) for cls in ("dog", "other")}


def prepare_transfer(archive, out=config.DATA, n_train=config.TRANSFER_TRAIN_PER_CLASS,
                     n_test=config.TRANSFER_TEST_PER_CLASS, size=config.TRANSFER_SIZE):
    """Part 1: a small balanced subset at 224x224, saved as PNGs in out/{train,test}/{dog,other}."""
    random.seed(0)
    with zipfile.ZipFile(archive) as z:
        for cls, files in _class_files(z).items():
            random.shuffle(files)
            picked = {"train": files[:n_train], "test": files[n_train:n_train + n_test]}
            for split, members in picked.items():
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for i, m in enumerate(members):
                    try:
                        img = Image.open(io.BytesIO(z.read(m))).convert("RGB")
                    except Exception as e:  # a few files in scraped datasets are broken
                        print("skip", m, e, file=sys.stderr)
                        continue
                    img.resize((size, size), Image.BICUBIC).save(d / f"{cls}_{i:04d}.png")
            print(f"  {cls}: {len(picked['train'])} train, {len(picked['test'])} test")


def _load_small(z, name, size):
    img = Image.open(io.BytesIO(z.read(name)))
    img.draft("RGB", (size * 4, size * 4))  # fast JPEG decode at reduced size
    img = img.convert("RGB")
    s = min(img.size)
    img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                    (img.width + s) // 2, (img.height + s) // 2))
    return np.asarray(img.resize((size, size), Image.BICUBIC))


def prepare_e2e(archive, out=config.E2E_DATA, n_test=config.E2E_TEST_PER_CLASS,
                n_pool=config.E2E_POOL_PER_CLASS, size=config.E2E_SIZE):
    """Part 2: the whole archive at 32x32, split into train / test / pool, saved as one .npz.

    pool = images the attacker draws base dogs from; never seen in train or test.
    """
    random.seed(0)
    splits = {k: ([], []) for k in ("train", "test", "pool")}
    with zipfile.ZipFile(archive) as z:
        for cls, files in _class_files(z).items():
            label = 1 if cls == "dog" else 0
            random.shuffle(files)
            parts = {"test": files[:n_test], "pool": files[n_test:n_test + n_pool],
                     "train": files[n_test + n_pool:]}
            for split, members in parts.items():
                for m in members:
                    try:
                        splits[split][0].append(_load_small(z, m, size))
                    except Exception:  # a few scraped files are broken
                        continue
                    splits[split][1].append(label)
    arrays = {}
    for k, (x, y) in splits.items():
        arrays[f"x_{k}"] = np.stack(x)
        arrays[f"y_{k}"] = np.array(y, dtype=np.int64)
        print(f"  {k}: {len(y)} images, {int(arrays[f'y_{k}'].sum())} dogs")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)


def transfer_split(split):
    """Part 1 image paths and labels (1 = dog, 0 = not dog) for "train" or "test"."""
    dogs = sorted((config.DATA / split / "dog").glob("*.png"))
    others = sorted((config.DATA / split / "other").glob("*.png"))
    if not dogs:
        raise SystemExit(f"No images in {config.DATA / split}. Run: python run.py prepare")
    return dogs + others, torch.tensor([1] * len(dogs) + [0] * len(others))


def load_e2e():
    """Part 2 tensors: {"train"|"test"|"pool": (images (N, 3, 32, 32) in [0, 1], labels)}."""
    if not config.E2E_DATA.exists():
        raise SystemExit(f"{config.E2E_DATA} is missing. Run: python run.py prepare")
    d = np.load(config.E2E_DATA)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255
    return {k: (to_t(d[f"x_{k}"]), torch.from_numpy(d[f"y_{k}"])) for k in ("train", "test", "pool")}
