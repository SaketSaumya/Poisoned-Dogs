# Poison Frogs: a hands-on replication

This project reproduces **"Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks"** (Shafahi et al., NeurIPS 2018) on a dog vs. not-dog image dataset.

The attack makes a classifier call **one chosen photo** a dog. It does this by adding a few dog pictures to the training data. Every poison looks like an ordinary dog and carries its honest label, "dog". The chosen photo itself is never touched.

Everything runs on a laptop. A GPU helps (Apple Silicon or CUDA) but is not required.

---

## Contents

1. [Quick start](#quick-start)
2. [The concepts](#the-concepts)
3. [Running each step](#running-each-step)
4. [What we found](#what-we-found)
5. [How the code is organised](#how-the-code-is-organised)
6. [Differences from the paper](#differences-from-the-paper)
7. [Notes and troubleshooting](#notes-and-troubleshooting)

---

## Quick start

```bash
pip install -r requirements.txt
python run.py all --quick        # about 15 minutes on a first run; drop --quick for the full run (about 1 hour)
```

Before running, you need two files:

- **`archive.zip`**: the dataset. Put it in this folder or one folder up, or pass `--archive PATH`. Any zip works if it has a `dog/` folder and an `other/` folder of images. We used a "Dog vs Not-Dog" collection of about 25,000 JPEGs (12.5k per class, 3 GB).
- **`me.jpg`**: the target, the image the attack should flip to "dog". We used a photo of the presenter. Any image the clean model calls "not dog" works. Use `--target PATH` to point somewhere else.

The first run downloads the ImageNet ResNet-50 weights from torchvision, about 100 MB.

Results land in `figures/` (charts and GIFs), `poisons/` (the poison images) and `data/` (datasets, caches and results).

---

## The concepts

### Training-time attacks

A classifier has two lives. During **training** it learns from labeled examples. At **test time** it is frozen and labels new images.

- **Evasion attacks**, also called adversarial examples, change the *test* image. The classic case is a panda photo with invisible noise added that the model calls a gibbon. This needs access to the input being judged.
- **Poisoning attacks** change the *training* data instead.

This paper is the hardest kind of poisoning:

- **Targeted:** one chosen test image is misclassified. Everything else behaves normally, so accuracy checks see nothing.
- **Clean-label:** the attacker never mislabels anything. Every poison really looks like a dog and is honestly labeled "dog", so a human reviewer sees nothing wrong.

### Features: what the network sees

A network turns an image into a list of numbers at its second-to-last layer, the **feature vector** `f(x)`. ResNet-50 produces 2,048 numbers. The last layer only ever sees these numbers and draws a boundary through them: dog on one side, not dog on the other.

Tiny pixel changes get amplified layer by layer. Two images that look identical to us can have very different features. That gap is what the attack exploits.

### Feature collision

The attack uses three images:

| Symbol | Image | In this project |
|---|---|---|
| `t` | **target**: the test image to misclassify | `me.jpg` |
| `b` | **base**: an ordinary image of the class we want, here a dog | the training dog closest to the target |
| `p` | **poison**: the base, carefully edited | `poisons/poison_*.png` |

The poison solves

```
p = argmin_x  ||f(x) − f(t)||²  +  β · ||x − b||²
```

- **First term, collide in feature space:** to the network, the poison should look like the target. `f(x) − f(t)` subtracts the two feature vectors number by number. `||…||²` squares the differences and adds them up.
- **Second term, stay close in pixel space:** to a human, the poison should still look like the base dog, so it gets labeled "dog". `x − b` is the per-pixel change.
- **β** sets the balance. Small β gives a tight collision but visible noise. Large β gives an invisible change but a weaker attack.

When the victim retrains on the poison, labeled "dog", the boundary moves to put the poison on the dog side. The target sits at almost the same spot in feature space, so it goes too.

### Algorithm 1: forward-backward splitting

`f` is a deep network, so the equation has no closed-form solution. The paper alternates two steps, starting from `x = b`:

```
forward:   x̂ = x − λ · ∇ₓ ||f(x) − f(t)||²       move the features toward the target's
backward:  x  = (x̂ + λβ · b) / (1 + λβ)           pull the pixels back toward the base
```

- `λ` is the step size.
- `∇ₓ` is the gradient with respect to the pixels. It is backpropagated through the **frozen** network: the image changes, not the network.
- `λβ` is how hard each round pulls back toward the dog. With λ = 0.005 and β = 10, each round blends the image 5% back toward the base.

Pixels are clipped to the valid range after each round, and poisons are saved as **PNG**. JPEG compression would wipe out the small changes.

### Part 1: transfer learning, the "one-shot kill"

Many real systems take a pretrained network, freeze it, and retrain only the last layer on their own data. The poison's features cannot change during retraining, and the last layer must fit every training point. So **one poison is enough**.

The paper ran this 1,099 times, attacking every test image once, and succeeded every time.

### Part 2: end-to-end training

If the victim retrains **every** layer, the early layers adjust. The poison only collided with the target because it exploited small quirks in those layers, and retraining fixes the quirks. The poison's features slide back among the dogs and the target stays put. **One poison fails.** The paper adds two tricks:

- **Watermarking:** before crafting, blend a faint copy of the target into each base, `b ← γ·t + (1 − γ)·b`, with opacity γ = 30% (20% also tested). The poison then shares real low-level features with the target that retraining cannot separate.
- **Many diverse poisons:** about 50 poisons from 50 different dogs. No single feature can separate all of them from the target, so retraining drags the target into the dog class along with them.

The paper's ablation finds that all three ingredients are needed: Algorithm 1, diverse bases and the watermark. With 50 poisons at 30% opacity it reaches about 60% success, and 70% on outlier targets.

### Threat model and limits

- **Attacker's knowledge:** the feature extractor's architecture and weights. That is realistic, since many systems start from public pretrained models.
- **Attacker's access:** a way to get images into the training set, for example a scraped dataset.
- **Targets one exact image.** A different photo of the same person has different features and would most likely not flip. We did not test this.
- **Watermarks can be visible.** At 32×32 pixels, or with a colourful target, a 30% watermark can be visible. The paper says it is unnoticeable "for some target instances", not all.

---

## Running each step

Every step is a subcommand of `run.py`. `python run.py <step> --help` lists its options with defaults.

| Step | What it does | Time* | Writes |
|---|---|---|---|
| `prepare` | Builds both datasets from the archive: 2,400 images at 224×224 for part 1, and all ~25k images at 32×32 for part 2 | ~1.5 min | `data/train`, `data/test`, `data/e2e_32.npz` |
| `transfer` | **Part 1.** Trains the clean model, crafts 5 poisons, retrains, and reports P(dog \| target) for 1 to 5 poisons | ~1 min | `poisons/`, `figures/1_…` to `4_…` |
| `animate` | **Part 1.** GIF of one poison being crafted: zoomed pixels, the change, the feature distance, and a 2-D map of feature space | ~1 min | `figures/pixel_shift.gif` |
| `beta-sweep` | **Part 1.** One poison per β from the same dog, side by side | ~2 min | `figures/beta_sweep.png` |
| `watermark-gif` | **Part 2.** GIF of the watermark blend, 0% to 100% opacity | ~5 s | `figures/watermark_demo.gif` |
| `e2e-pretrain` | **Part 2.** Trains the clean 32×32 AlexNet (cached afterwards) | ~5 min | `data/e2e_clean_alexnet.pt` |
| `e2e-samples` | **Part 2.** Grid of the 32×32 training images | ~5 s | `figures/e2e_32_samples.png` |
| `e2e-attack` | **Part 2.** One end-to-end attack with N watermarked poisons | ~40 s | (prints the result) |
| `e2e-ablation` | **Part 2.** Success rate vs. number of poisons and opacity; 75 full retrains by default | 40–70 min | `data/e2e_ablation.json`, `figures/e2e_ablation.png` |
| `all` | Every step above, in order (`--quick`: a 6-run ablation) | ~1 h (`--quick`: ~15 min) | everything |

\*On an Apple M-series laptop. Slower without a GPU.

### Useful variations

```bash
# a different target image
python run.py transfer --target path/to/photo.jpg

# one poison only
python run.py transfer --n-poisons 1
```

```bash
# the poison with no pull back toward the dog: it dissolves into colour noise
python run.py animate --beta 0 --lr 0.08 --steps 2000 --amplify 1 --out pixel_shift_large.gif

# the same, showing just the dog photo
python run.py animate --beta 0 --lr 0.08 --steps 2000 --image-only --out pixel_shift_large_image.gif
```

```bash
# the betas where the change is visible
python run.py beta-sweep --betas 0 0.5 1 2 5 10 --out beta_sweep_visible.png
```

```bash
# end to end: does one poison work? (it shouldn't)
python run.py e2e-attack --n-poisons 1

# 50 poisons with no watermark
python run.py e2e-attack --n-poisons 50 --opacity 0

# redraw the ablation chart from saved results, no training
python run.py e2e-ablation --plot-only
```

`poison_frogs_demo.ipynb` walks through part 1 cell by cell using the same code, which is handy for presenting.

---

## What we found

### Part 1: transfer learning (frozen ResNet-50)

| | The paper | Our run |
|---|---|---|
| Network | InceptionV3, frozen | ResNet-50, frozen |
| Task | dog vs. fish (ImageNet) | dog vs. not-dog |
| Targets | 1,099 test images, one at a time | 1 photo |
| Result | 100% success, 1 poison each | P(dog \| target): 0% → 91–96% with 1 poison, 98–99% with 5 |
| Accuracy change | −0.2% on average | +0.2% (98.8% → 99.0%) |
| Biggest pixel change | not reported | about 12–24 out of 255, varying by run (38–43 dB PSNR) |

Five other faces from the test set stay at 0% dog, so the attack is targeted.

### Part 2: end-to-end (32×32 AlexNet, all layers retrained)

The clean model scores 89.4% test accuracy and gives the target 0.7% dog. The table shows successes out of 5 trials, where success means P(dog | target) > 50% after retraining:

| Poisons | No watermark | 20% watermark | 30% watermark |
|---|---|---|---|
| 1 | 0/5 | 0/5 | 0/5 |
| 5 | 0/5 | 0/5 | 0/5 |
| 10 | 0/5 | 0/5 | 1/5 |
| 25 | 0/5 | 5/5 | 5/5 |
| 50 | 2/5 | 5/5 | 5/5 |

The pattern matches the paper: one poison never works, success rises with the number of poisons, and the watermark matters most. Test accuracy stayed between 89.1% and 89.6% throughout.

Our rates are higher than the paper's 60%, because our setting is easier: one fixed target, two classes and only 5 trials per cell. One difference: without a watermark, 50 poisons still won 2 of 5 trials, whereas the paper's ablation found the attack fails.

---

## How the code is organised

```
run.py                   one command per step (argument parsing only)
poison_frogs_demo.ipynb  part 1 as a notebook, for presenting
poisonfrogs/
  config.py              paths and default settings, all in one place
  images.py              load / save / compare images (PSNR, biggest pixel change)
  data.py                build both datasets from the archive
  features.py            frozen ResNet-50 feature extractor + trainable last layer
  attack.py              Algorithm 1 (craft) and watermarking
  transfer.py            part 1 experiments: setup, the attack, the beta sweep
  e2e.py                 part 2: small AlexNet, pretraining, full retraining, attack, ablation
  plots.py               static figures
  animations.py          GIFs
  gif.py                 flicker-free GIF writer (one shared palette)
```

Suggested reading order: `attack.py` (the whole attack in about 40 lines), then `transfer.py`, then `e2e.py`.

---

## Differences from the paper

- **Network and data:** part 1 uses ResNet-50 instead of InceptionV3, and dog vs. not-dog instead of dog vs. fish. Part 2 uses the paper's AlexNet (Appendix F, Table 1) with 2 outputs instead of 10.
- **Forward step:** uses Adam instead of plain gradient descent. It converges more reliably without hand-tuning λ, which the paper does not report. The backward step is the paper's.
- **β scale:** our β is on a pixel scale of 0 to 1, with the pull per step being λβ. It is not directly comparable to the paper's β₀ = 0.25, which the paper scales by image size as β = β₀ · 2048² / dim_b².
- **Base selection:** part 1 uses the dogs whose features are already closest to the target. Part 2 uses random dogs from a held-out pool, as the paper does.
- **Pretraining:** the part 2 clean model trains for 60 epochs with flip/shift augmentation; the paper uses 200. Without augmentation, the small network called a bearded face 99% dog. The retraining that matters for the attack uses the paper's exact settings: Adam, learning rate 1.85e-5, 10 epochs, batch 128.

---

## Notes and troubleshooting

- **Numbers vary a little between runs.** The optimisation oscillates slightly near the end, and GPU arithmetic is not bit-exact. Expect a few percentage points of difference, for example 91% or 96% with one poison.
- **Caches:** `data/features_resnet50.pt` and `data/e2e_clean_alexnet.pt` are reused once they exist. Delete them, or use `e2e-pretrain --force`, after changing the data.
- **`prepare` is deterministic:** the same archive always gives the same splits. The defaults for `watermark-gif` (`dog_0163.png` and the parrot `other_0028.png`) and the comparison faces in `3_targeted.png` refer to images in those splits.
- **Long runs:** the ablation saves after every retrain, so a stopped run keeps what it finished.
- **Privacy:** `me.jpg`, `data/`, `poisons/` and `figures/` contain the target photo or images derived from it. Leave them out when sharing the project.

## Reference

Ali Shafahi, W. Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, Tom Goldstein. *Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks.* NeurIPS 2018.
