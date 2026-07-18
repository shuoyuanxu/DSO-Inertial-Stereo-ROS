#!/usr/bin/env python3
"""GitHub-ready GIFs: 3D sparse map with the trajectory drawn INSIDE it,
from several viewpoints. Renders PNG frames; ffmpeg palettes them into GIFs."""
import numpy as np, os, sys, glob, shutil, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

CAP = sys.argv[1]                                  # capture dir (rich_stereo)
OUT = sys.argv[2]                                  # output dir for gifs
TMP = "/tmp/gifframes"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(f"{CAP}/frame_*.npz"))
good = []
for f in files:
    try:
        np.load(f)["t"]; good.append(f)
    except Exception:
        pass
files = good
print("frames:", len(files))

def load(f):
    d = np.load(f)
    return d["cloud"], d["path"]

# camera frame (X right, Y down, Z forward) -> display (forward, lateral, up)
def disp(a):
    if len(a) == 0: return np.zeros((0, 3))
    return np.column_stack([a[:, 2], a[:, 0], -a[:, 1]])

cloud_f, path_f = load(files[-1])
C = disp(cloud_f); P = disp(path_f)
# clip outliers for a clean scene
if len(C):
    lo, hi = np.percentile(C, [1.0, 99.0], axis=0)
    C = C[np.all((C >= lo) & (C <= hi), axis=1)]
# subsample for render speed / gif size
if len(C) > 18000:
    C = C[np.random.default_rng(0).choice(len(C), 18000, replace=False)]
print("scene points:", len(C), "path points:", len(P))

lims = np.vstack([C, P]) if len(P) else C
xl = np.percentile(lims[:, 0], [0.5, 99.5]); yl = np.percentile(lims[:, 1], [0.5, 99.5])
zl = np.percentile(lims[:, 2], [0.5, 99.5])

def new_ax(figsize=(8, 6)):
    fig = plt.figure(figsize=figsize, facecolor='#111111')
    # axes fill the figure — matplotlib's default 3d margins waste most of the frame
    ax = fig.add_axes([-0.06, -0.10, 1.12, 1.16], projection='3d', facecolor='#111111')
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
    # true proportions in the ground plane, vertical exaggerated so the 60 m
    # corridor doesn't render as a flat pancake
    try: ax.set_box_aspect((xl.ptp(), yl.ptp(), max(zl.ptp(), 1) * 3.0))
    except Exception: pass
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False; a.pane.set_edgecolor('#333333')
        a.line.set_color('#444444'); a.label.set_color('#aaaaaa')
    ax.tick_params(colors='#777777', labelsize=7)
    ax.set_xlabel('forward (m)', fontsize=8); ax.set_ylabel('lateral (m)', fontsize=8)
    ax.set_zlabel('up (m)', fontsize=8)
    return fig, ax

def draw(ax, C, P, title):
    if len(C):
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=1.3, c=C[:, 2], cmap='viridis',
                   alpha=0.75, linewidths=0, rasterized=True)
    if len(P) > 1:
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color='#ff4d4d', lw=2.5, label='trajectory')
        ax.scatter([P[-1, 0]], [P[-1, 1]], [P[-1, 2]], color='#ffdd00', s=45, depthshade=False)
    ax.get_figure().text(0.5, 0.955, title, color='#eeeeee', fontsize=12, ha='center')
    if len(P) > 1:
        lg = ax.legend(loc='upper right', fontsize=8, facecolor='#222222', edgecolor='#444444')
        for t in lg.get_texts(): t.set_color('#eeeeee')

def to_gif(name, fps=15, width=820):
    pal = f"{TMP}/pal.png"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{TMP}/f_%04d.png",
        "-vf",f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",pal],check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{TMP}/f_%04d.png","-i",pal,
        "-lavfi",f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        "-loop","0",f"{OUT}/{name}"],check=True)
    sz = os.path.getsize(f"{OUT}/{name})".rstrip(")"))/1e6
    print(f"  {name}  {sz:.1f} MB")

def reset_tmp():
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)

# ---------------------------------------------------------------- 1. ORBIT
print("rendering orbit...")
reset_tmp()
N = 72
for i in range(N):
    fig, ax = new_ax()
    draw(ax, C, P, "Stereo DSO — sparse map + trajectory")
    ax.view_init(elev=22, azim=-60 + 360.0 * i / N)
    fig.savefig(f"{TMP}/f_{i:04d}.png", dpi=90, facecolor=fig.get_facecolor(),
                )
    plt.close(fig)
to_gif("stereo_orbit.gif")

# ------------------------------------------------------------- 2. BUILD-UP
print("rendering build-up (oblique)...")
reset_tmp()
sel = files[::max(1, len(files)//70)]
for i, f in enumerate(sel):
    cf, pf = load(f)
    Ci, Pi = disp(cf), disp(pf)
    if len(Ci) > 14000:
        Ci = Ci[np.random.default_rng(1).choice(len(Ci), 14000, replace=False)]
    fig, ax = new_ax()
    draw(ax, Ci, Pi, "Stereo DSO — map building live")
    ax.view_init(elev=24, azim=-65)
    fig.savefig(f"{TMP}/f_{i:04d}.png", dpi=90, facecolor=fig.get_facecolor(),
                )
    plt.close(fig)
to_gif("stereo_building.gif", fps=12)

# ------------------------------------------------- 3. FIXED ANGLE SHOWCASE
for name, elev, azim, title in [
        ("stereo_topdown.gif", 89, -90, "Stereo DSO — top-down"),
        ("stereo_side.gif",     6, -90, "Stereo DSO — side view"),
        ("stereo_chase.gif",   14, -35, "Stereo DSO — oblique")]:
    print("rendering", name, "...")
    reset_tmp()
    for i, f in enumerate(sel):
        cf, pf = load(f)
        Ci, Pi = disp(cf), disp(pf)
        if len(Ci) > 14000:
            Ci = Ci[np.random.default_rng(2).choice(len(Ci), 14000, replace=False)]
        fig, ax = new_ax()
        draw(ax, Ci, Pi, title)
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(f"{TMP}/f_{i:04d}.png", dpi=90, facecolor=fig.get_facecolor(),
                    )
        plt.close(fig)
    to_gif(name, fps=12)

shutil.rmtree(TMP, ignore_errors=True)
print("done ->", OUT)
