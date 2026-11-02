"""Loading, saving and comparing images. Images are float tensors (3, H, W) in [0, 1]."""
import math

import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


def load_image(path, size):
    """Centre-crop to a square, resize to size x size."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(img.size)
    img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                    (img.width + s) // 2, (img.height + s) // 2))
    return TF.to_tensor(img.resize((size, size), Image.BICUBIC))


def save_png(x, path):
    # PNG is lossless: JPEG compression would wipe out a small perturbation
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(x.cpu().clamp(0, 1)).save(path)


def quantize(x):
    """Round to 8-bit levels, as the image would be after saving."""
    return (x * 255).round() / 255


def psnr(a, b):
    """Peak signal-to-noise ratio in dB; above ~40 dB a change is invisible."""
    mse = ((a - b) ** 2).mean().item()
    return 10 * math.log10(1 / mse) if mse > 0 else float("inf")


def max_change(a, b):
    """Biggest single pixel change, out of 255."""
    return int(((a - b).abs().max() * 255).round())


def to_numpy(x):
    """(3, H, W) tensor -> (H, W, 3) array for matplotlib."""
    return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
