#!/usr/bin/env python3
"""Combined showcase GIF (no titles):
   top    = 3D sparse map + trajectory (chase view)
   bottom = raw undistorted frame  |  featured frame (selected points)"""
import numpy as np, os, sys, glob, shutil, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

CAP = sys.argv[1]
OUTGIF = sys.argv[2]
TMP = "/tmp/comboframes"

files = sorted(glob.glob(f"{CAP}/frame_*.npz"))
good = []
for f in files:
    try:
        np.load(f)["t"]; good.append(f)
    except Exception:
        pass
files = good
print("frames:", len(files))

def disp(a):   # camera frame (X right, Y down, Z fwd) -> (forward, lateral, up)
    if len(a) == 0: return np.zeros((0, 3))
    return np.column_stack([a[:, 2], a[:, 0], -a[:, 1]])

# scene extent from the final frame
d = np.load(files[-1])
C = disp(d["cloud"]); P = disp(d["path"])
if len(C):
    lo, hi = np.percentile(C, [1.0, 99.0], axis=0)
    C = C[np.all((C >= lo) & (C <= hi), axis=1)]
lims = np.vstack([C, P]) if len(P) else C
XL = np.percentile(lims[:, 0], [0.5, 99.5]); YL = np.percentile(lims[:, 1], [0.5, 99.5])
ZL = np.percentile(lims[:, 2], [0.5, 99.5])

shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
sel = files[::max(1, len(files) // 80)]
rng = np.random.default_rng(0)

for i, f in enumerate(sel):
    d = np.load(f)
    Ci, Pi = disp(d["cloud"]), disp(d["path"])
    if len(Ci) > 15000:
        Ci = Ci[rng.choice(len(Ci), 15000, replace=False)]
    img, raw = d["img"], d["raw"]

    fig = plt.figure(figsize=(9, 8.2), facecolor='#111111')
    # 3D axes are deliberately oversized: with the axis off, matplotlib's large
    # 3d margins would otherwise leave the scene tiny inside its cell.
    ax = fig.add_axes([-0.30, 0.15, 1.60, 0.92], projection='3d', facecolor='#111111')
    ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
    try: ax.set_box_aspect((XL.ptp(), YL.ptp(), max(ZL.ptp(), 1) * 3.0))
    except Exception: pass
    ax.set_axis_off()
    if len(Ci):
        ax.scatter(Ci[:, 0], Ci[:, 1], Ci[:, 2], s=1.3, c=Ci[:, 2], cmap='viridis',
                   vmin=ZL[0], vmax=ZL[1],
                   alpha=0.75, linewidths=0, rasterized=True)
    if len(Pi) > 1:
        ax.plot(Pi[:, 0], Pi[:, 1], Pi[:, 2], color='#ff4d4d', lw=2.6)
        ax.scatter([Pi[-1, 0]], [Pi[-1, 1]], [Pi[-1, 2]], color='#ffdd00', s=48, depthshade=False)
    ax.view_init(elev=14, azim=-35)

    # --- bottom left: raw undistorted frame
    axr = fig.add_axes([0.005, 0.005, 0.492, 0.30]); axr.set_facecolor('#111111'); axr.axis('off')
    if raw.size: axr.imshow(raw, cmap='gray', aspect='auto')
    # --- bottom right: featured frame (selected points)
    axf = fig.add_axes([0.503, 0.005, 0.492, 0.30]); axf.set_facecolor('#111111'); axf.axis('off')
    if img.size: axf.imshow(img, aspect='auto')

    fig.savefig(f"{TMP}/f_{i:04d}.png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)

pal = f"{TMP}/pal.png"
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{TMP}/f_%04d.png",
    "-vf","fps=12,scale=760:-1:flags=lanczos,palettegen=stats_mode=diff",pal], check=True)
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{TMP}/f_%04d.png","-i",pal,
    "-lavfi","fps=12,scale=760:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
    "-loop","0",OUTGIF], check=True)
shutil.rmtree(TMP, ignore_errors=True)
print(f"wrote {OUTGIF}  {os.path.getsize(OUTGIF)/1e6:.1f} MB")
