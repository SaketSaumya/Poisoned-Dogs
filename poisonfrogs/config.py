"""Paths and default settings shared by every step.

beta follows the paper's Algorithm 1: each backward step is
x = (x_hat + lr*beta*b) / (1 + lr*beta), so the pull toward the base per step is lr*beta.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
POISONS = ROOT / "poisons"
DEFAULT_TARGET = ROOT / "me.jpg"

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def find_archive():
    """The dataset zip, looked for next to the project and one folder up."""
    for p in (ROOT / "archive.zip", ROOT.parent / "archive.zip"):
        if p.exists():
            return p
    return ROOT / "archive.zip"


# --- part 1: transfer learning (paper Sec. 3) -----------------------------------
TRANSFER_SIZE = 224
TRANSFER_TRAIN_PER_CLASS = 900      # same 900 per class as the paper's dog-vs-fish set
TRANSFER_TEST_PER_CLASS = 300
TRANSFER_N_POISONS = 5
TRANSFER_BETA = 10.0                # lr * beta = 0.05 pull toward the base per step
TRANSFER_LR = 0.005
TRANSFER_STEPS = 1000               # paper: maxIters = 1000
HEAD_EPOCHS = 2000                  # last layer trained to convergence (full batch)
FEATURES_CACHE = DATA / "features_resnet50.pt"
# human faces among the not-dog test images (picked by eye), used to show the attack is targeted
COMPARE_INDICES = (5, 19, 26, 32, 34)

# --- part 2: end-to-end training (paper Sec. 4) ---------------------------------
E2E_SIZE = 32                       # CIFAR-10 sized, like the paper
E2E_TEST_PER_CLASS = 1500
E2E_POOL_PER_CLASS = 1000           # attacker's base images, never in train/test
E2E_DATA = DATA / "e2e_32.npz"
E2E_WEIGHTS = DATA / "e2e_clean_alexnet.pt"
E2E_ABLATION = DATA / "e2e_ablation.json"
E2E_PRETRAIN_EPOCHS = 60
E2E_RETRAIN_LR = 1.85e-5            # paper: Adam at 1.85e-5 ...
E2E_RETRAIN_EPOCHS = 10             # ... over 10 epochs ...
E2E_BATCH = 128                     # ... with batch size 128
E2E_BETA = 5.0                      # lr * beta = 0.05, same pull as part 1
E2E_LR = 0.01
E2E_STEPS = 2000
E2E_OPACITY = 0.3                   # paper's main watermark setting

# --- watermark animation ----------------------------------------------------------
WATERMARK_BASE = DATA / "train" / "dog" / "dog_0163.png"     # the white dog used in the other demos
WATERMARK_TARGET = DATA / "test" / "other" / "other_0028.png"  # a parrot; the paper's figure uses a bird
