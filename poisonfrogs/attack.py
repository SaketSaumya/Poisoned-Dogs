"""The attack itself: Algorithm 1 (feature collision) and watermarking.

Poison p = argmin_x ||f(x) - f(t)||^2 + beta * ||x - b||^2
    t = target (the image to misclassify), b = base (a dog), f = the victim's feature extractor.
"""
import torch

from . import config
from .images import quantize


def watermark(bases, target, opacity):
    """Paper Sec. 4.2: blend a faint copy of the target into each base, b <- opacity*t + (1-opacity)*b."""
    return opacity * target + (1 - opacity) * bases


def craft(features, bases, target, beta, lr, steps, on_step=None):
    """Algorithm 1, forward-backward splitting, on a batch of bases at once.

    Forward step:  move x to reduce ||f(x) - f(t)||^2 (the gradient flows through the
                   frozen network to the pixels; we use Adam here, the paper uses plain
                   gradient descent).
    Backward step: x = (x_hat + lr*beta*b) / (1 + lr*beta), a pull back toward the base.

    beta is one number, or one per base (to compare several betas in one run).
    on_step(step, x) is called before the first step and after every step.
    Returns the poisons rounded to 8-bit pixel values, on the CPU.
    """
    b = bases.to(config.DEVICE)
    with torch.no_grad():
        f_t = features(target.to(config.DEVICE))
    pull = lr * torch.as_tensor(beta, dtype=torch.float32, device=config.DEVICE).reshape(-1, 1, 1, 1)
    x = b.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    if on_step:
        on_step(0, x.detach())
    for step in range(1, steps + 1):
        opt.zero_grad()
        ((features(x) - f_t) ** 2).sum().backward()
        opt.step()                                                   # forward step
        with torch.no_grad():
            x.copy_(((x + pull * b) / (1 + pull)).clamp(0, 1))       # backward step
        if on_step:
            on_step(step, x.detach())
    return quantize(x.detach()).cpu()
