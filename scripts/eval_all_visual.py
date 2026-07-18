#!/usr/bin/env python3
"""Evaluate every visual algorithm in the GT folder + our modes, same protocol."""
import numpy as np, os

SP = '/tmp/claude-1000/-home-shu-catkin-DSO-ws-src/898b3da5-2545-42c7-8230-164fb71df7e1/scratchpad'
B = '/media/shu/disk_1tb/DatasetPaperReview/NOV/easy_AprilAdd'
g = np.genfromtxt(f'{B}/refined_odometry.csv', delimiter=',', skip_header=1)
GTt, GTp = g[:, 2], g[:, 5:8]
GT_SPAN = GTt[-1] - GTt[0]

def um(A, B_, ws):
    ca, cb = A.mean(0), B_.mean(0); A0, B0 = A - ca, B_ - cb
    H = A0.T @ B0 / len(A); U, D, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ S @ Vt
    s = (D @ S.diagonal()) / ((B0**2).sum() / len(B0)) if ws else 1.0
    return s, R, ca - s * (R @ cb)

def tum(fn):
    d = np.genfromtxt(fn, comments='#')
    return d[:, :4]

def ev(traj, metric):
    gi = np.searchsorted(GTt, traj[:, 0]); gi = np.clip(gi, 1, len(GTt) - 1)
    pk = np.where(np.abs(GTt[gi] - traj[:, 0]) < np.abs(GTt[gi-1] - traj[:, 0]), gi, gi-1)
    ok = np.abs(GTt[pk] - traj[:, 0]) < 0.2
    if ok.sum() < 30: return None
    E, G = traj[ok, 1:4], GTp[pk[ok]]
    ss, Rs, ts = um(E, G, True); Gs = (ss * (Rs @ G.T)).T + ts
    r = {'n': int(ok.sum()),
         'sim3': np.sqrt((np.linalg.norm(E - Gs, axis=1)**2).mean()),
         'cov': 100.0 * (traj[ok, 0].max() - traj[ok, 0].min()) / GT_SPAN}
    if metric:
        se, Re, te = um(E, G, False); Ge = (se * (Re @ G.T)).T + te
        r['se3'] = np.sqrt((np.linalg.norm(E - Ge, axis=1)**2).mean())
        r['scale'] = 1 / ss
    else:
        r['se3'] = None; r['scale'] = None
    return r

# name, path, metric?, family
ROWS = [
    ("OURS stereo (gated)",   f"{SP}/m_stereo.npy",     True,  "ours"),
    ("OURS mono",             f"{SP}/m_mono.npy",       False, "ours"),
    ("OURS mono_imu",         f"{SP}/m_mono_imu.npy",   True,  "ours"),
    ("OURS stereo_imu",       f"{SP}/m_stereo_imu.npy", True,  "ours"),
    ("OpenVINS mono-inertial",   f"{B}/openvins_mi.txt",   True,  "base"),
    ("OpenVINS stereo-inertial", f"{B}/openvins_si.txt",   True,  "base"),
    ("ORB-SLAM3 mono",           f"{B}/orb3_m.txt",        False, "base"),
    ("ORB-SLAM3 mono-inertial",  f"{B}/orb3_mi.txt",       True,  "base"),
    ("ORB-SLAM3 stereo",         f"{B}/orb3_s.txt",        True,  "base"),
    ("VINS-Fusion mono-inertial",   f"{B}/vins_fusion_mi.txt", True, "base"),
    ("VINS-Fusion stereo",          f"{B}/vins_fusion_s.txt",  True, "base"),
    ("VINS-Fusion stereo-inertial", f"{B}/vins_fusion_si.txt", True, "base"),
    ("DSO (upstream, mono)",        f"{B}/dso.txt",            False, "base"),
]

out = []
print("%-30s %6s %9s %9s %8s %8s" % ("system", "pairs", "SE3", "Sim3", "scale", "cov%"))
for name, path, metric, fam in ROWS:
    if not os.path.exists(path):
        print("%-30s  <missing>" % name); continue
    traj = np.load(path)[:, :4] if path.endswith('.npy') else tum(path)
    r = ev(traj, metric)
    if r is None:
        print("%-30s  <too few GT matches>" % name); continue
    se3 = f"{r['se3']:.2f}" if r['se3'] is not None else "  --"
    sc = f"{r['scale']:.2f}" if r['scale'] is not None else "  --"
    print("%-30s %6d %9s %8.2f %8s %7.0f" % (name, r['n'], se3, r['sim3'], sc, r['cov']))
    out.append((name, fam, metric, r))
np.save(f"{SP}/all_visual_eval.npy", np.array(out, dtype=object), allow_pickle=True)
