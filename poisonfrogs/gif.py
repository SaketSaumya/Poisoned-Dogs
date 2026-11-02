"""Flicker-free GIF writer.

A GIF frame holds at most 256 colours. Matplotlib's writer picks a new palette for
every frame, so pixels that never change still shift colour from frame to frame.
Here every frame is encoded with ONE shared palette and no dithering instead.
"""
import numpy as np
from PIL import Image


def save_gif(fig, draw, n_frames, out, fps=12):
    """Call draw(k) for each frame k, render the figure, and write all frames to out."""
    frames = []
    for k in range(n_frames):
        draw(k)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    # build the palette from a sample of frames stacked together, so it covers every colour used
    sample = frames[::max(1, n_frames // 8)] + [frames[-1]]
    w, h = sample[0].size
    sheet = Image.new("RGB", (w, h * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * h))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    # no dithering: identical input pixels must map to identical output pixels in every frame
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=round(1000 / fps), loop=0)
