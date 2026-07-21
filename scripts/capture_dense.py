#!/usr/bin/env python3
"""Capture DepthMsg frames for offline GIF rendering.

Mirrors polytunnel_vio/scripts/capture_rich.py: subscribe, snapshot at a fixed
rate into npz files, render later. Keeping capture and render separate means a
slow matplotlib pass never stalls the live pipeline.

Usage: capture_dense.py <out_dir> [rate_hz]
"""
import os
import sys
import threading

import numpy as np
import rospy
from cv_bridge import CvBridge
from polytunnel_vio.msg import DepthMsg

OUT = sys.argv[1]
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
os.makedirs(OUT, exist_ok=True)

bridge = CvBridge()
lock = threading.Lock()
latest = {"msg": None}
count = [0]


def cb(m):
    depth = bridge.imgmsg_to_cv2(m.depth, "passthrough").astype(np.float32)
    conf = bridge.imgmsg_to_cv2(m.confidence, "passthrough").astype(np.float32)
    img = bridge.imgmsg_to_cv2(m.image, "mono8")
    with lock:
        latest["msg"] = (m.image.header.stamp.to_sec(), img.copy(),
                         depth.copy(), conf.copy(), m.backend,
                         np.array(m.camToWorld), np.array(m.Intrinsics))


def snap(_):
    with lock:
        if latest["msg"] is None:
            return
        t, img, depth, conf, backend, c2w, K = latest["msg"]
    i = count[0]
    count[0] += 1
    np.savez_compressed(os.path.join(OUT, "frame_%04d.npz" % i),
                        t=t, img=img, depth=depth, conf=conf,
                        backend=backend, camToWorld=c2w, K=K)


rospy.init_node("capture_dense", anonymous=True)
rospy.Subscriber("/dense_depth/depth_info", DepthMsg, cb, queue_size=4)
rospy.Timer(rospy.Duration(1.0 / RATE), snap)
rospy.loginfo("capture_dense -> %s at %.1f Hz", OUT, RATE)
rospy.spin()
print("captured %d frames to %s" % (count[0], OUT))
