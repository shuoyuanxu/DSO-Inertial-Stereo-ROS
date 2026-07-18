#!/usr/bin/env python3
"""Benchmark summary: accuracy (log ATE) + sequence coverage for every visual system."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/shu/catkin_DSO_ws/readme_assets/benchmark.png'
# (name, Sim3 ATE m, coverage %, ours?, diverged?)
D = [
 ("OURS mono",                  0.23,  66, True,  False),
 ("ORB-SLAM3 mono",             0.31,  78, False, False),
 ("DSO (upstream) mono",        0.31,  98, False, False),
 ("ORB-SLAM3 stereo",           0.52,  25, False, False),
 ("OpenVINS mono-inertial",     0.62, 100, False, False),
 ("ORB-SLAM3 mono-inertial",    0.80,  99, False, False),
 ("OURS stereo (range-gated)",  0.92,  95, True,  False),
 ("OURS stereo-inertial",       0.97,   1, True,  False),
 ("VINS-Fusion stereo-inertial",3.06, 100, False, False),
 ("VINS-Fusion mono-inertial", 15.76, 100, False, False),
 ("OURS mono-inertial",        21.34,   6, True,  False),
 ("OpenVINS stereo-inertial", 40111.0, 100, False, True),
 ("VINS-Fusion stereo",     29662557.0,100, False, True),
]
D = sorted(D, key=lambda r: r[1])
names = [d[0] for d in D]; ate = np.array([d[1] for d in D])
cov = np.array([d[2] for d in D]); ours = [d[3] for d in D]; div = [d[4] for d in D]
y = np.arange(len(D))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#141414',
                              gridspec_kw={'width_ratios': [2.1, 1]})
for a in (ax, ax2):
    a.set_facecolor('#141414'); a.tick_params(colors='#bbbbbb', labelsize=9)
    for sp in a.spines.values(): sp.set_color('#444444')
    a.grid(alpha=0.15, color='#666666', axis='x')

cols = ['#4da3ff' if o else ('#aa4444' if d else '#888888') for o, d in zip(ours, div)]
ax.barh(y, ate, color=cols, height=0.7)
ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels(names, color='#eeeeee', fontsize=9)
ax.invert_yaxis(); ax.set_xlabel('Sim3 ATE (m, log scale) — lower is better', color='#dddddd')
ax.set_title('Trajectory accuracy', color='#ffffff', fontsize=12)
for i, (v, d) in enumerate(zip(ate, div)):
    ax.text(v * 1.25, i, "diverged" if d else f"{v:.2f} m",
            va='center', color='#ff8888' if d else '#dddddd', fontsize=8)

ax2.barh(y, cov, color=['#4da3ff' if o else '#888888' for o in ours], height=0.7)
ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.invert_yaxis()
ax2.set_xlim(0, 108); ax2.set_xlabel('sequence tracked (%) — higher is better', color='#dddddd')
ax2.set_title('Robustness / coverage', color='#ffffff', fontsize=12)
for i, v in enumerate(cov):
    ax2.text(v + 2, i, f"{v:.0f}%", va='center', color='#dddddd', fontsize=8)

fig.suptitle('Visual odometry benchmark — polytunnel sequence vs tag-map ground truth\n'
             'blue = this repo · grey = baselines · red = diverged',
             color='#ffffff', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT, dpi=110, facecolor=fig.get_facecolor())
print("wrote", OUT)
