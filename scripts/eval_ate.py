#!/usr/bin/env python3
"""ATE evaluation of a trajectory against ground truth.

Usage:
    eval_ate.py <trajectory> <groundtruth> [max_dt]

<trajectory>   .npy array of rows [t, x, y, z]  (from tools/dump_paths.py)
               or a TUM-format text file  [t x y z qx qy qz qw]
<groundtruth>  rostopic-echo CSV of a nav_msgs/Odometry topic
               (e.g. refined_odometry.csv: stamp in col 2, position in cols 5-7)
               or a TUM-format text file
Reports SE3 ATE (honest metric test), Sim3 ATE (shape test) with the residual
alignment scale, and path-length ratio.
"""
import numpy as np
import sys


def load(fn):
    if fn.endswith('.npy'):
        return np.load(fn)[:, :4]
    d = np.genfromtxt(fn, comments='#', delimiter=',' if fn.endswith('.csv') else None)
    if fn.endswith('.csv'):
        return np.column_stack([d[:, 2], d[:, 5], d[:, 6], d[:, 7]])
    return d[:, :4]


def umeyama(A, B, with_scale):
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0 / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = (D @ S.diagonal()) / ((B0**2).sum() / len(B0)) if with_scale else 1.0
    return s, R, ca - s * (R @ cb)


def main():
    est, gt = load(sys.argv[1]), load(sys.argv[2])
    max_dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    gi = np.searchsorted(gt[:, 0], est[:, 0])
    gi = np.clip(gi, 1, len(gt) - 1)
    pick = np.where(np.abs(gt[gi, 0] - est[:, 0]) < np.abs(gt[gi - 1, 0] - est[:, 0]), gi, gi - 1)
    ok = np.abs(gt[pick, 0] - est[:, 0]) < max_dt
    E, G = est[ok, 1:4], gt[pick[ok], 1:4]
    print("matched pairs:", ok.sum())

    for label, ws in [("SE3 (metric)", False), ("Sim3 (shape)", True)]:
        s, R, t = umeyama(E, G, ws)
        Ga = (s * (R @ G.T)).T + t
        err = np.linalg.norm(E - Ga, axis=1)
        extra = ", align scale %.3f" % (1 / s) if ws else ""
        print("%s: ATE RMSE %.2f m, median %.2f, max %.2f%s" %
              (label, np.sqrt((err**2).mean()), np.median(err), err.max(), extra))

    le = np.linalg.norm(np.diff(E, axis=0), axis=1).sum()
    lg = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
    print("path length: est %.1f m vs GT %.1f m (ratio %.3f)" % (le, lg, le / lg))


if __name__ == "__main__":
    main()
