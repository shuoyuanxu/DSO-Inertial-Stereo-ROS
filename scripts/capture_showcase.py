#!/usr/bin/env python3
"""Capture the dense showcase stream: fused cloud + trajectory + both images.

One run per depth backend; make_showcase_gif.py pairs the two runs by timestamp
and renders the 2x2 panel. Same capture/render split as capture_rich.py so a
slow matplotlib pass never stalls the live pipeline.

Usage: capture_showcase.py <out_dir> [rate_hz]
"""
import os
import sys
import threading

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path

OUT = sys.argv[1]
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
os.makedirs(OUT, exist_ok=True)

lock = threading.Lock()
state = {"cloud": None, "rgb": None, "sparse": None,
         "path": None, "raw": None, "feat": None}
count = [0]


def cloud_cb(m):
    """Fused dense cloud: XYZ + packed RGB from dense_mapping."""
    if m.point_step < 12:
        return
    n = m.width * m.height
    buf = np.frombuffer(m.data, np.uint8).reshape(n, m.point_step)
    xyz = buf[:, :12].copy().view(np.float32).reshape(n, 3)
    off = {f.name: f.offset for f in m.fields}
    rgb = None
    if "rgb" in off:
        c = buf[:, off["rgb"]:off["rgb"] + 3].astype(np.uint8)
        rgb = c[:, ::-1]                       # PCL packs BGR
    ok = np.isfinite(xyz).all(1)
    with lock:
        state["cloud"] = xyz[ok]
        state["rgb"] = rgb[ok] if rgb is not None else None


def sparse_cb(m):
    """Sparse DSO map (vi_dso/cloud): XYZ only, world frame, backend-independent.

    Overlaid on the dense cloud in the GIF so you can see the DSO skeleton the
    dense reconstruction is built on top of. Same world frame as the dense
    cloud, so no transform is needed.
    """
    if m.point_step < 12:
        return
    n = m.width * m.height
    if n == 0:
        return
    xyz = np.frombuffer(m.data, np.uint8).reshape(n, m.point_step)[:, :12] \
        .copy().view(np.float32).reshape(n, 3)
    with lock:
        state["sparse"] = xyz[np.isfinite(xyz).all(1)]


def path_cb(m):
    a = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                  for p in m.poses])
    with lock:
        state["path"] = a


def _img(m):
    if m.encoding == "mono8":
        return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
    if m.encoding in ("bgr8", "rgb8"):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        return a[:, :, ::-1] if m.encoding == "bgr8" else a
    return None


def raw_cb(m):
    a = _img(m)
    if a is not None:
        with lock:
            state["raw"] = a.copy()


def feat_cb(m):
    a = _img(m)
    if a is not None:
        with lock:
            state["feat"] = a.copy()


def snap(_):
    with lock:
        if state["cloud"] is None or state["raw"] is None:
            return
        cloud = state["cloud"]
        rgb = state["rgb"]
        sparse = state["sparse"] if state["sparse"] is not None else np.zeros((0, 3), np.float32)
        path = state["path"] if state["path"] is not None else np.zeros((0, 3))
        raw = state["raw"]
        feat = state["feat"] if state["feat"] is not None else np.zeros((0, 0, 3), np.uint8)
    i = count[0]
    count[0] += 1
    np.savez_compressed(
        os.path.join(OUT, "frame_%04d.npz" % i),
        t=rospy.Time.now().to_sec(), cloud=cloud,
        rgb=rgb if rgb is not None else np.zeros((0, 3), np.uint8),
        sparse=sparse, path=path, raw=raw, feat=feat)


rospy.init_node("capture_showcase", anonymous=True)
rospy.Subscriber("/dense_mapping/cloud", PointCloud2, cloud_cb, queue_size=1)
rospy.Subscriber("/vi_dso/cloud", PointCloud2, sparse_cb, queue_size=1)
rospy.Subscriber("/vi_dso/path", Path, path_cb, queue_size=2)
rospy.Subscriber("/vi_dso/image_raw", Image, raw_cb, queue_size=2)
rospy.Subscriber("/vi_dso/image_points", Image, feat_cb, queue_size=2)
rospy.Timer(rospy.Duration(1.0 / RATE), snap)
rospy.loginfo("capture_showcase -> %s at %.1f Hz", OUT, RATE)
rospy.spin()
print("captured %d frames to %s" % (count[0], OUT))
