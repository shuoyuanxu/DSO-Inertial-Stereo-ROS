#!/usr/bin/env python3
"""Capture selected-points image + point cloud + path during a run, time-stamped,
so a composite video can be rendered offline. Saves to <out>/rich_<tag>/."""
import rospy, os, sys, threading
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path

OUT = sys.argv[1]           # output dir for this mode
IMG_TOPIC   = sys.argv[2] if len(sys.argv) > 2 else "/vi_dso/image_points"
CLOUD_TOPIC = sys.argv[3] if len(sys.argv) > 3 else "/vi_dso/cloud"
PATH_TOPIC  = sys.argv[4] if len(sys.argv) > 4 else "/vi_dso/path"
RAW_TOPIC   = sys.argv[5] if len(sys.argv) > 5 else "/vi_dso/image_raw"
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock()
state = {"img": None, "cloud": None, "path": None, "raw": None}
frames = []                 # list of (t, img_png_idx, cloud_idx)
img_count = [0]

def img_cb(m):
    if m.encoding not in ("bgr8", "rgb8"): return
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    if m.encoding == "bgr8": a = a[:, :, ::-1]
    with lock: state["img"] = (m.header.stamp.to_sec(), a.copy())

def raw_cb(m):
    if m.encoding == "mono8":
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
    elif m.encoding in ("bgr8", "rgb8"):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "bgr8": a = a[:, :, ::-1]
    else:
        return
    with lock: state["raw"] = a.copy()

def cloud_cb(m):
    if m.point_step < 12: return
    pts = np.frombuffer(m.data, np.float32).reshape(-1, m.point_step // 4)[:, :3]
    with lock: state["cloud"] = (m.header.stamp.to_sec(), pts.copy())

def path_cb(m):
    a = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in m.poses])
    with lock: state["path"] = a

def snap(_):
    with lock:
        if state["img"] is None: return
        t = state["img"][0]
        img = state["img"][1]
        cloud = state["cloud"][1] if state["cloud"] else np.zeros((0, 3))
        path = state["path"] if state["path"] is not None else np.zeros((0, 3))
        raw = state["raw"] if state["raw"] is not None else np.zeros((0, 0), np.uint8)
    i = img_count[0]; img_count[0] += 1
    np.savez_compressed(f"{OUT}/frame_{i:04d}.npz", t=t, img=img, cloud=cloud, path=path, raw=raw)

rospy.init_node("rich_capture", anonymous=True)
rospy.Subscriber(IMG_TOPIC, Image, img_cb, queue_size=2)
rospy.Subscriber(CLOUD_TOPIC, PointCloud2, cloud_cb, queue_size=2)
rospy.Subscriber(PATH_TOPIC, Path, path_cb, queue_size=2)
rospy.Subscriber(RAW_TOPIC, Image, raw_cb, queue_size=2)
rospy.Timer(rospy.Duration(0.4), snap)   # ~2.5 fps capture
rospy.spin()
print("captured", img_count[0], "frames to", OUT)
