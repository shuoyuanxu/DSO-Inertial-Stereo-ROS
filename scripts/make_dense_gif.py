#!/usr/bin/env python3
"""GitHub-ready GIF of the dense depth stream: keyframe | depth | confidence.

Same pipeline as polytunnel_vio/scripts/make_gifs.py - render PNGs, let ffmpeg
palette them - so the output matches the existing README assets in look and size.

Usage: make_dense_gif.py <capture_dir> <out_dir> [name.gif]
"""
import os
import sys
import glob
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAP = sys.argv[1]
OUT = sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else None
TMP = "/tmp/denseframes"
FPS = 8
WIDTH = 900

os.makedirs(OUT, exist_ok=True)
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

files = sorted(glob.glob(os.path.join(CAP, "frame_*.npz")))
good = []
for f in files:
    try:
        np.load(f)["depth"]
        good.append(f)
    except Exception:
        pass                      # a frame written while we were reading it
files = good
if not files:
    sys.exit("no frames in %s" % CAP)

backend = str(np.load(files[0])["backend"])
if NAME is None:
    NAME = "dense_%s.gif" % backend
print("frames: %d  backend: %s" % (len(files), backend))

# Fix the depth colour scale across the whole run, otherwise every frame
# renormalises and the GIF flickers in a way that hides real changes.
sample = np.concatenate([
    np.load(f)["depth"][np.load(f)["depth"] > 0].ravel()
    for f in files[::max(1, len(files) // 30)]])
vmin, vmax = np.percentile(sample, [2, 98])
print("depth scale: %.2f - %.2f m" % (vmin, vmax))

for i, f in enumerate(files):
    d = np.load(f)
    img, depth, conf = d["img"], d["depth"], d["conf"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), facecolor="#111111")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("keyframe", color="#dddddd", fontsize=11)

    dm = np.ma.masked_where(depth <= 0, depth)
    im = axes[1].imshow(dm, cmap="turbo", vmin=vmin, vmax=vmax)
    valid = 100.0 * (depth > 0).mean()
    axes[1].set_title("depth  %.1f-%.1f m   (valid %.0f%%)" % (vmin, vmax, valid),
                      color="#dddddd", fontsize=11)

    axes[2].imshow(conf, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("confidence  (mean %.2f)" % conf.mean(),
                      color="#dddddd", fontsize=11)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#333333")

    fig.suptitle("%s   frame %d/%d" % (backend, i + 1, len(files)),
                 color="#888888", fontsize=10, y=0.99)
    fig.savefig(os.path.join(TMP, "f_%04d.png" % i),
                facecolor="#111111", dpi=80)
    plt.close(fig)

    if (i + 1) % 25 == 0:
        print("  rendered %d/%d" % (i + 1, len(files)))

pal = os.path.join(TMP, "pal.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(TMP, "f_%04d.png"),
                "-vf", "fps=%d,scale=%d:-1:flags=lanczos,palettegen=stats_mode=diff"
                % (FPS, WIDTH), pal], check=True)
out = os.path.join(OUT, NAME)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(TMP, "f_%04d.png"),
                "-i", pal, "-lavfi",
                "fps=%d,scale=%d:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3"
                % (FPS, WIDTH), "-loop", "0", out], check=True)
print("%s  %.1f MB" % (out, os.path.getsize(out) / 1e6))
