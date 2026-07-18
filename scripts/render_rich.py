#!/usr/bin/env python3
"""Render a composite video from captured frames: selected-points image (left),
point cloud top-down (middle), trajectory vs GT (right)."""
import numpy as np, os, sys, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

CAP = sys.argv[1]                 # capture dir
OUTMP4 = sys.argv[2]              # output mp4
TITLE = sys.argv[3] if len(sys.argv) > 3 else ""
GT = '/media/shu/disk_1tb/DatasetPaperReview/NOV/easy_AprilAdd/refined_odometry.csv'

files = sorted(glob.glob(f"{CAP}/frame_*.npz"))
if not files:
    print("no frames in", CAP); sys.exit(1)
# drop any corrupt (partially-written) frames
good = []
for f in files:
    try:
        np.load(f)["t"]; good.append(f)
    except Exception:
        pass
files = good

# GT for the trajectory panel (aligned to the final path via Sim3 for display)
g = np.genfromtxt(GT, delimiter=',', skip_header=1); GTp = g[:, 5:8]

# Camera/world frame: X=right(lateral), Y=down(vertical), Z=forward(vehicle motion).
# Robot bird's-eye (top-down) = forward Z on the horizontal axis, lateral X on the
# vertical axis (looking straight down the -Y/up axis).
allc = []
last_path = np.zeros((0, 3))
for f in files:
    d = np.load(f)
    if len(d["cloud"]): allc.append(d["cloud"])
    if len(d["path"]) > len(last_path): last_path = d["path"]
allc = np.vstack(allc) if allc else np.zeros((1, 3))
cz = np.percentile(allc[:, 2], [1, 99])   # forward extent
cx = np.percentile(allc[:, 0], [1, 99])   # lateral extent

# image panel on the left (its natural 4:3), map + trajectory stacked on the right
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
                      left=0.03, right=0.98, top=0.90, bottom=0.07, wspace=0.15, hspace=0.28)
axi = fig.add_subplot(gs[:, 0])
axc = fig.add_subplot(gs[0, 1])
axt = fig.add_subplot(gs[1, 1])
fig.suptitle(TITLE, fontsize=14)
writer = FFMpegWriter(fps=8, bitrate=3000)
with writer.saving(fig, OUTMP4, dpi=80):
    for f in files:
        d = np.load(f)
        img, cloud, path = d["img"], d["cloud"], d["path"]
        axi.clear(); axc.clear(); axt.clear()
        # left: selected-points image
        axi.imshow(img); axi.set_title("selected points (inv-depth colour)"); axi.axis('off')
        # top-right: sparse map, robot bird's-eye (forward Z ->, lateral X up), colour = height
        if len(cloud):
            axc.scatter(cloud[:, 2], cloud[:, 0], s=1, c=-cloud[:, 1], cmap='viridis')
        axc.set_title("sparse map (top-down: forward →)"); axc.set_aspect('equal')
        axc.set_xlim(cz); axc.set_ylim(cx); axc.grid(alpha=0.3)
        axc.set_xlabel("forward Z (m)"); axc.set_ylabel("lateral X (m)")
        # bottom-right: trajectory, same bird's-eye
        if len(path):
            axt.plot(path[:, 2], path[:, 0], '-', color='tab:blue', lw=2, label='estimate')
            axt.plot(path[-1, 2], path[-1, 0], 'o', color='red')
        axt.set_title("trajectory (top-down: forward →)"); axt.set_aspect('equal')
        axt.grid(alpha=0.3); axt.legend(loc='upper right')
        axt.set_xlabel("forward Z (m)"); axt.set_ylabel("lateral X (m)")
        writer.grab_frame()
plt.close(fig)
print("wrote", OUTMP4, "from", len(files), "frames")
