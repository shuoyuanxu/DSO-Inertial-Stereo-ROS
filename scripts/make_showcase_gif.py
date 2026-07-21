#!/usr/bin/env python3
"""Dense showcase GIF, 2x2:

    top-left      trajectory inside the STEREO-generated dense cloud
    top-right     trajectory inside the MVSNET-generated dense cloud
    bottom-left   raw undistorted frame DSO sees
    bottom-right  feature / selected-points image

The two clouds come from two separate runs of the same bag, so frames are
paired by elapsed time within each run rather than by index - the backends
run at different rates (stereo ~11 Hz, unimvsnet ~5 Hz) and would otherwise
drift apart down the sequence.

Usage: make_showcase_gif.py <stereo_cap> <mvsnet_cap> <out_dir> [name.gif]
"""
import os
import sys
import glob
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

STEREO_CAP = sys.argv[1]
MVS_CAP = sys.argv[2]
OUT = sys.argv[3]
NAME = sys.argv[4] if len(sys.argv) > 4 else "dense_showcase.gif"
TMP = "/tmp/showcaseframes"
FPS = 6          # input sequence rate; NOT an output resample (see encode below)
WIDTH = 780
KEEP_EVERY = 2   # capture runs at 2.5 Hz, so consecutive frames barely differ
COLORS = 128     # palette cap; together with KEEP_EVERY keeps the GIF ~8 MB

os.makedirs(OUT, exist_ok=True)
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)


def load_run(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "frame_*.npz"))):
        try:
            z = np.load(f)
            z["cloud"]
            out.append(f)
        except Exception:
            pass                     # written while we were reading it
    if not out:
        sys.exit("no frames in %s" % d)
    t0 = float(np.load(out[0])["t"])
    times = np.array([float(np.load(f)["t"]) - t0 for f in out])
    return out, times


sfiles, stimes = load_run(STEREO_CAP)
mfiles, mtimes = load_run(MVS_CAP)
print("stereo frames %d (%.0f s)  mvsnet frames %d (%.0f s)"
      % (len(sfiles), stimes[-1], len(mfiles), mtimes[-1]))

# drive the animation off whichever run is shorter, so neither pane freezes
N = min(len(sfiles), len(mfiles))
drive = stimes if len(sfiles) <= len(mfiles) else mtimes
drive = drive[:N]


# camera frame (X right, Y down, Z forward) -> display (forward, lateral, up),
# matching make_gifs.py so this sits next to the existing README assets
def disp(a):
    if len(a) == 0:
        return np.zeros((0, 3))
    return np.column_stack([a[:, 2], a[:, 0], -a[:, 1]])


def final_extent(files):
    z = np.load(files[-1])
    return disp(z["cloud"]), disp(z["path"])


Cs, Ps = final_extent(sfiles)
Cm, Pm = final_extent(mfiles)
allpts = np.vstack([p for p in (Cs, Cm, Ps, Pm) if len(p)])
xl = np.percentile(allpts[:, 0], [0.5, 99.5])
yl = np.percentile(allpts[:, 1], [0.5, 99.5])
zl = np.percentile(allpts[:, 2], [1.0, 99.0])
print("scene extent  x %.1f..%.1f  y %.1f..%.1f  z %.1f..%.1f"
      % (xl[0], xl[1], yl[0], yl[1], zl[0], zl[1]))


def draw_cloud(ax, cloud, rgb, path, title, npts=20000, xlim=None):
    """Colour by the cloud's own RGB when it has any, else by height.

    This used to be height-only on purpose: DSO tracks on greyscale, so every
    fused point came back r=g=b and a dim polytunnel rendered as near-black
    points on a dark background. dense_depth now resamples real colour into the
    keyframe, so the true colour is available and reads the scene far better
    than a turbo ramp. The height fallback stays for captures recorded before
    that, and for runs with ~colorize false.
    """
    ax.set_facecolor("#111111")
    C = disp(cloud)
    col = np.asarray(rgb) if rgb is not None else np.zeros((0, 3))
    have_rgb = len(col) == len(C) and len(col) > 0 and col.ptp() > 0
    if len(C) > npts:
        idx = np.random.default_rng(0).choice(len(C), npts, replace=False)
        C = C[idx]
        if have_rgb:
            col = col[idx]
    if len(C):
        if have_rgb:
            # brighten: the tunnel is dim and a dark background eats it
            c = np.clip(col.astype(np.float32) / 255.0 * 1.35, 0, 1)
            ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=1.4, c=c,
                       marker=".", linewidths=0, depthshade=False)
        else:
            ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=1.4, c=C[:, 2],
                       cmap="turbo", vmin=zl[0], vmax=zl[1],
                       marker=".", linewidths=0, depthshade=False)
    P = disp(path)
    if len(P):
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#ff2200", linewidth=2.6)
        ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], s=45, c="#ffdd33", marker="o",
                   edgecolors="#000000", linewidths=0.5, depthshade=False)
    # Chase the camera: with the axis fixed to the full 111 m corridor, an
    # early frame (a few metres of map) renders as a dot in an empty box. A
    # forward window that follows the current position keeps the cloud filling
    # the panel the whole way down the tunnel. Lateral/vertical stay fixed -
    # they are small and constant, so only the forward axis needs to move.
    xw = xl if xlim is None else np.asarray(xlim)
    ax.set_xlim(xw); ax.set_ylim(yl); ax.set_zlim(zl)
    # true ground-plane proportions, vertical exaggerated so the corridor
    # does not render as a flat pancake (same trick as make_gifs.py)
    try:
        ax.set_box_aspect((xw.ptp(), yl.ptp() * 2.2, max(zl.ptp(), 1) * 2.6))
    except Exception:
        pass
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#2a2a2a")
        a.line.set_color("#2a2a2a")
        a.set_ticklabels([])
    ax.set_title(title, color="#dddddd", fontsize=11, pad=-4)
    ax.view_init(elev=16, azim=-78)
    ax.dist = 7.2          # pull the camera in; matplotlib's default wastes the frame


for i in range(N):
    t = drive[i]
    si = int(np.argmin(np.abs(stimes - t)))
    mi = int(np.argmin(np.abs(mtimes - t)))
    zs = np.load(sfiles[si])
    zm = np.load(mfiles[mi])

    # shared forward window centred just behind the current camera, so both
    # panels stay comparable and the accumulated trail is what fills the frame
    tips = [disp(z["path"])[-1, 0] for z in (zs, zm) if len(z["path"])]
    fwd = max(tips) if tips else xl[1]
    SPAN, AHEAD = 34.0, 6.0                     # metres of trail, metres ahead
    lo = max(xl[0], fwd - (SPAN - AHEAD))
    xwin = np.array([lo, min(xl[1], lo + SPAN)])

    fig = plt.figure(figsize=(12, 8.4), facecolor="#111111")

    # explicit axes rects: matplotlib's 3d subplots leave most of their cell
    # empty, which on a 2x2 grid wastes half the GIF
    ax1 = fig.add_axes([-0.02, 0.47, 0.54, 0.46], projection="3d",
                       facecolor="#111111")
    draw_cloud(ax1, zs["cloud"], zs["rgb"], zs["path"],
               "trajectory in STEREO dense cloud  (%d pts)" % len(zs["cloud"]),
               xlim=xwin)

    ax2 = fig.add_axes([0.48, 0.47, 0.54, 0.46], projection="3d",
                       facecolor="#111111")
    draw_cloud(ax2, zm["cloud"], zm["rgb"], zm["path"],
               "trajectory in MVSNET dense cloud  (%d pts)" % len(zm["cloud"]),
               xlim=xwin)

    ax3 = fig.add_axes([0.045, 0.02, 0.42, 0.44])
    raw = zm["raw"]
    ax3.imshow(raw, cmap="gray" if raw.ndim == 2 else None, aspect="equal")
    ax3.set_title("raw frame", color="#dddddd", fontsize=11, pad=4)

    ax4 = fig.add_axes([0.535, 0.02, 0.42, 0.44])
    feat = zm["feat"]
    if feat.size:
        ax4.imshow(feat, cmap="gray" if feat.ndim == 2 else None, aspect="equal")
    ax4.set_facecolor("#111111")
    ax4.set_title("selected points (inverse depth)", color="#dddddd",
                  fontsize=11, pad=4)

    for a in (ax3, ax4):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#333333")

    fig.text(0.5, 0.985, "polytunnel dense reconstruction   t = %.0f s" % t,
             color="#777777", fontsize=10, ha="center")
    fig.savefig(os.path.join(TMP, "f_%04d.png" % i), facecolor="#111111", dpi=80)
    plt.close(fig)

    if (i + 1) % 20 == 0:
        print("  rendered %d/%d" % (i + 1, N))

# Encode with -framerate on the INPUT, and NO fps filter.
#
# ffmpeg reads an image sequence at 25 fps unless told otherwise, so a
# "fps=7" output filter RESAMPLES it: 164 rendered frames become 46 and
# most of the animation is silently discarded. Setting the input rate keeps
# every frame; size is controlled by decimating frames explicitly (KEEP_EVERY)
# and by capping the palette, not by dropping frames after the fact.
seq = os.path.join(TMP, "f_%04d.png")
if KEEP_EVERY > 1:
    sub = os.path.join(TMP, "sub")
    os.makedirs(sub, exist_ok=True)
    for j, k in enumerate(range(0, N, KEEP_EVERY)):
        os.symlink(os.path.join(TMP, "f_%04d.png" % k),
                   os.path.join(sub, "g_%04d.png" % j))
    seq = os.path.join(sub, "g_%04d.png")
    print("decimated to %d frames (every %d)" % (len(range(0, N, KEEP_EVERY)), KEEP_EVERY))

pal = os.path.join(TMP, "pal.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", seq,
                "-vf", "scale=%d:-1:flags=lanczos,palettegen=max_colors=%d:stats_mode=diff"
                % (WIDTH, COLORS), pal], check=True)
out = os.path.join(OUT, NAME)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", seq, "-i", pal, "-lavfi",
                "scale=%d:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4"
                % WIDTH, "-loop", "0", out], check=True)

# verify what actually landed in the file, not what we intended
probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=nb_frames", "-of", "csv=p=0", out],
                       capture_output=True, text=True).stdout.strip()
print("%s  %.1f MB  %s frames" % (out, os.path.getsize(out) / 1e6, probe))
