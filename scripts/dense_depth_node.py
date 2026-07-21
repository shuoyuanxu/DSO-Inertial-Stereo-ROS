#!/usr/bin/env python3
"""Dense depth for DSO sliding windows.

Subscribes  vi_dso/sliding_window  (dso_dense_msgs/SlidingWindowsMsg)
Publishes   dense_depth/depth_info (dso_dense_msgs/DepthMsg)

The backend is chosen with the ~backend param ("unimvsnet" or "stereo") and is
the only thing that differs between the two arms of the benchmark - the message
in and the message out are identical, so dense_mapping cannot tell them apart.

Inference is slower than DSO produces keyframes, so windows are handled by a
worker thread behind a bounded queue. Under load the OLDEST pending window is
dropped, never the newest: a reconstruction wants the freshest geometry, and
silently falling further behind real time is worse than skipping a keyframe.
Every drop is counted and reported.
"""
import os
import sys
import threading
import collections
import time

import numpy as np
import rospy
import rospkg
from cv_bridge import CvBridge
from polytunnel_vio.msg import SlidingWindowsMsg, DepthMsg

# the MVS engine lives in include/MVS, alongside the DSO core in include/
sys.path.insert(0, os.path.join(
    rospkg.RosPack().get_path("polytunnel_vio"), "include"))

from MVS.window import WindowAssembler  # noqa: E402


class DenseDepthNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.assembler = WindowAssembler()

        pkg = rospkg.RosPack().get_path("polytunnel_vio")
        self.backend_name = rospy.get_param("~backend", "unimvsnet")

        # A plane sweep over views that share a camera centre returns confident
        # nonsense, and DSO's first windows are exactly that. Gate on the actual
        # geometry rather than hoping.
        self.min_views = int(rospy.get_param("~min_views", 3))
        self.min_baseline = float(rospy.get_param("~min_baseline", 0.05))

        self.queue_len = int(rospy.get_param("~queue_len", 2))
        self._pending = collections.deque(maxlen=self.queue_len)
        self._cv = threading.Condition()
        self._running = True

        self.n_recv = 0        # complete windows assembled
        self.n_done = 0        # windows actually reconstructed
        self.n_skipped = 0     # rejected by the geometry gate
        self.n_dropped = 0     # evicted from the queue while busy
        self._times = collections.deque(maxlen=50)

        self.backend = self._make_backend(pkg)
        self.colorizer = self._make_colorizer(pkg)

        self.pub = rospy.Publisher("dense_depth/depth_info", DepthMsg, queue_size=10)
        self.sub = rospy.Subscriber("vi_dso/sliding_window", SlidingWindowsMsg,
                                    self.on_window, queue_size=200)

        self.worker = threading.Thread(target=self._loop)
        self.worker.daemon = True
        self.worker.start()
        rospy.Timer(rospy.Duration(10.0), self._report)

    def _make_backend(self, pkg):
        if self.backend_name == "unimvsnet":
            from MVS.unimvsnet_backend import UniMVSNetBackend
            ckpt = rospy.get_param(
                "~ckpt", os.path.join(pkg, "models", "unimvsnet_blendedmvs.ckpt"))
            b = UniMVSNetBackend(
                ckpt=ckpt,
                max_w=int(rospy.get_param("~max_w", 640)),
                max_h=int(rospy.get_param("~max_h", 480)),
                numdepth=int(rospy.get_param("~numdepth", 192)),
                depth_mode=rospy.get_param("~depth_mode", "unification"),
                fp16=bool(rospy.get_param("~fp16", False)))
            rospy.loginfo("unimvsnet: %s | %s", os.path.basename(ckpt), b.load_report)
            return b
        if self.backend_name == "stereo":
            from MVS.stereo_backend import StereoBackend
            return StereoBackend.from_params(pkg)
        raise ValueError("unknown ~backend %r (expected unimvsnet or stereo)"
                         % self.backend_name)

    def _make_colorizer(self, pkg):
        """Optional: resample raw colour into the keyframe so the cloud isn't gray.

        DSO hands us a mono8 keyframe (it converts to grayscale before tracking),
        so without this every fused point gets r=g=b. Failing to build it is not
        fatal - the pipeline just stays grayscale, as before.
        """
        if not bool(rospy.get_param("~colorize", True)):
            return None
        try:
            from MVS.colorizer import Colorizer
            from MVS.stereo_backend import read_dso_calib
            calib_dir = rospy.get_param(
                "~calib_dir", os.path.join(pkg, "calib", "polytunnel"))
            _, K, D, size = read_dso_calib(
                rospy.get_param("~calib_left",
                                os.path.join(calib_dir, "cam0.txt")))
            return Colorizer(
                K_raw=K, D=D, raw_size=size,
                topic=rospy.get_param("~left_topic",
                                      "/forwardLeft/image_raw/compressed"),
                max_time_diff=float(rospy.get_param("~color_time_diff", 0.05)))
        except Exception as e:
            rospy.logwarn("colorize disabled (%s); cloud will be grayscale", e)
            return None

    def on_window(self, msg):
        w = self.assembler.add(msg)
        if w is None:
            return
        self.n_recv += 1

        if len(w) < self.min_views:
            self.n_skipped += 1
            return
        if w.baseline_span() < self.min_baseline:
            # DSO init: all views effectively at one point
            self.n_skipped += 1
            rospy.logdebug("window %d: baseline %.4f m below %.4f, skipped",
                           w.msg_id, w.baseline_span(), self.min_baseline)
            return

        with self._cv:
            if len(self._pending) == self._pending.maxlen:
                self.n_dropped += 1     # deque discards the oldest on append
            self._pending.append(w)
            self._cv.notify()

    def _loop(self):
        while self._running and not rospy.is_shutdown():
            with self._cv:
                while not self._pending:
                    self._cv.wait(0.2)
                    if not self._running or rospy.is_shutdown():
                        return
                w = self._pending.popleft()
            try:
                self._process(w)
            except Exception as e:                       # keep the node alive
                rospy.logerr("window %d failed: %s", w.msg_id, e)
                import traceback
                rospy.logerr(traceback.format_exc())

    def _images(self, w):
        """Window images as HxWx3 float32 in [0,1]."""
        out = []
        for v in w.views:
            cv = self.bridge.imgmsg_to_cv2(v.image, "bgr8")
            out.append(np.asarray(cv, dtype=np.float32) / 255.0)
        return out

    def _process(self, w):
        t0 = time.time()
        res = self.backend.run(w, self._images(w))
        dt = time.time() - t0
        self._times.append(dt)
        self.n_done += 1

        ref = w.views[res.ref_index]
        finite = np.isfinite(res.depth) & (res.depth > 0)

        m = DepthMsg()
        # Colour is sampled on the depth map's own grid (res.K, res.depth shape),
        # not the keyframe's, so it stays aligned even when the backend resized.
        m.image = ref.image
        if self.colorizer is not None:
            rgb = self.colorizer.lookup(
                ref.image.header.stamp.to_sec(), res.K,
                (res.depth.shape[1], res.depth.shape[0]))
            if rgb is not None:
                m.image = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
                m.image.header = ref.image.header
        m.camToWorld = ref.camToWorld
        # K from the backend, not the message: the backend may have resized
        m.Intrinsics = [float(res.K[0, 0]), float(res.K[1, 1]),
                        float(res.K[0, 2]), float(res.K[1, 2])]
        m.depth = self.bridge.cv2_to_imgmsg(res.depth, encoding="32FC1")
        m.depth.header = ref.image.header
        m.confidence = self.bridge.cv2_to_imgmsg(res.confidence, encoding="32FC1")
        m.confidence.header = ref.image.header
        m.backend = res.backend
        self.pub.publish(m)

        rospy.logdebug("window %d: %.3f s, valid %.1f%%, conf mean %.3f",
                       w.msg_id, dt, 100.0 * finite.mean(),
                       float(np.mean(res.confidence)))

    def _report(self, _):
        if not self._times:
            rospy.loginfo("dense_depth[%s]: no windows processed yet "
                          "(recv %d, skipped %d)",
                          self.backend_name, self.n_recv, self.n_skipped)
            return
        t = np.array(self._times)
        rospy.loginfo("dense_depth[%s]: done %d | skipped %d | dropped %d | "
                      "%.2f s/window (%.2f Hz)",
                      self.backend_name, self.n_done, self.n_skipped,
                      self.n_dropped, t.mean(), 1.0 / max(t.mean(), 1e-6))

    def shutdown(self):
        self._running = False
        with self._cv:
            self._cv.notify_all()


if __name__ == "__main__":
    rospy.init_node("dense_depth")
    node = DenseDepthNode()
    rospy.on_shutdown(node.shutdown)
    rospy.loginfo("dense_depth up, backend=%s", node.backend_name)
    rospy.spin()
