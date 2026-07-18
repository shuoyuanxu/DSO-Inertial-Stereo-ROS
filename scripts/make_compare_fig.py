#!/usr/bin/env python3
"""2x2 visual comparison of the four modes against tag-map ground truth."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = '/tmp/claude-1000/-home-shu-catkin-DSO-ws-src/898b3da5-2545-42c7-8230-164fb71df7e1/scratchpad'
OUT = '/home/shu/catkin_DSO_ws/readme_assets/mode_comparison.png'
GT = '/media/shu/disk_1tb/DatasetPaperReview/NOV/easy_AprilAdd/refined_odometry.csv'
g = np.genfromtxt(GT, delimiter=',', skip_header=1); GTt = g[:, 2]; GTp = g[:, 5:8]

def um(A, B, ws):
    ca, cb = A.mean(0), B.mean(0); A0, B0 = A - ca, B - cb
    H = A0.T @ B0 / len(A); U, D, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ S @ Vt
    s = (D @ S.diagonal()) / ((B0**2).sum() / len(B0)) if ws else 1.0
    return s, R, ca - s * (R @ cb)

MODES = [("stereo",     "STEREO (range-gated)",        True,  0,   "#4da3ff"),
         ("mono",       "MONO (vision only)",          False, 0,   "#ffa64d"),
         ("mono_imu",   "MONO-INERTIAL (tight)",       True,  6,   "#ff5c5c"),
         ("stereo_imu", "STEREO-INERTIAL (tight)",     True,  569, "#7bd47b")]

fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor='#141414')
for ax, (tag, title, metric, resets, col) in zip(axes.ravel(), MODES):
    e = np.load(f"{SP}/m_{tag}.npy")
    gi = np.searchsorted(GTt, e[:, 0]); gi = np.clip(gi, 1, len(GTt) - 1)
    pk = np.where(np.abs(GTt[gi] - e[:, 0]) < np.abs(GTt[gi-1] - e[:, 0]), gi, gi-1)
    ok = np.abs(GTt[pk] - e[:, 0]) < 0.2
    E, G = e[ok, 1:4], GTp[pk[ok]]
    ss, Rs, ts = um(E, G, True); Gs = (ss * (Rs @ G.T)).T + ts
    sim3 = np.sqrt((np.linalg.norm(E - Gs, axis=1)**2).mean())
    if metric:
        se, Re, te = um(E, G, False); D = (se * (Re @ G.T)).T + te
        se3 = np.sqrt((np.linalg.norm(E - D, axis=1)**2).mean())
        lab = f"SE3 {se3:.2f} m · scale {1/ss:.2f}"
    else:
        D = Gs; lab = f"Sim3 {sim3:.2f} m (scale-free)"
    ax.set_facecolor('#141414')
    ax.plot(D[:, 0], D[:, 1], '-', color='#888888', lw=2.2, label='ground truth')
    ax.plot(E[:, 0], E[:, 1], '-', color=col, lw=2.0, label='estimate')
    ax.set_aspect('equal'); ax.grid(alpha=0.15, color='#666666')
    ax.set_title(f"{title}\n{lab}   ·   {resets} resets", color='#eeeeee', fontsize=11)
    ax.tick_params(colors='#888888', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#444444')
    lg = ax.legend(loc='best', fontsize=8, facecolor='#222222', edgecolor='#444444')
    for t in lg.get_texts(): t.set_color('#dddddd')

fig.suptitle("Mode comparison vs tag-map ground truth (polytunnel sequence)",
             color='#ffffff', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=110, facecolor=fig.get_facecolor())
print("wrote", OUT)
