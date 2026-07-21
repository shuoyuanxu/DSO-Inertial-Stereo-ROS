"""Colour for DSO keyframes.

DSO is a direct method and throws colour away at the first callback: the raw
frame is converted to MONO8 before it ever reaches the tracker, so the keyframe
that comes back out in SlidingWindowsMsg is grayscale and every point in the
fused cloud ends up r=g=b. Nothing downstream can recover it - the colour has to
be resampled from the raw camera stream.

The keyframe is a *pinhole* image that DSO produced by `crop`-undistorting the
EquiDistant fisheye, and the message carries the K it ended up with. So the map
back to the raw image is exactly the undistort map DSO itself used:

    keyframe pixel --K_dso^-1--> ray --equidistant distortion--> raw pixel

which is what cv2.fisheye.initUndistortRectifyMap(K_raw, D, I, K_dso, size)
builds. R = identity because `crop` only rescales the pinhole, it does not
rotate the camera. The map depends only on K_dso and the output size, both of
which are constant for a run, so it is built once and cached.

Colour is sampled from the *raw* topic rather than image_rect_color because the
rect topic is rectified with ROS' own intrinsics, which are not DSO's.
"""
import threading

import numpy as np
import cv2
import rospy


class Colorizer(object):
    """Buffers raw colour frames and resamples them into the DSO keyframe."""

    def __init__(self, K_raw, D, raw_size, topic, max_time_diff=0.05,
                 buffer_len=200):
        self.K_raw = np.asarray(K_raw, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64).reshape(4, 1)
        self.raw_size = (int(raw_size[0]), int(raw_size[1]))
        self.max_time_diff = float(max_time_diff)

        self._lock = threading.Lock()
        self._buf = {}
        self._buffer_len = int(buffer_len)
        self._map_key = None
        self._map = None

        self.n_hit = 0
        self.n_miss = 0

        from sensor_msgs.msg import CompressedImage, Image
        if topic.endswith("/compressed"):
            rospy.Subscriber(topic, CompressedImage, self._on_compressed,
                             queue_size=30)
        else:
            rospy.Subscriber(topic, Image, self._on_raw, queue_size=30)
        rospy.loginfo("colorizer: sampling %s (raw %dx%d)",
                      topic, self.raw_size[0], self.raw_size[1])

    def _store(self, stamp, img):
        with self._lock:
            self._buf[stamp] = img
            if len(self._buf) > self._buffer_len:
                for k in sorted(self._buf)[:len(self._buf) - self._buffer_len]:
                    del self._buf[k]

    def _on_compressed(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self._store(msg.header.stamp.to_sec(), img)

    def _on_raw(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._store(msg.header.stamp.to_sec(), img.copy())

    def _maps(self, K_dso, size):
        """keyframe pixel -> raw fisheye pixel. Cached; K and size are fixed."""
        key = (tuple(np.asarray(K_dso).ravel()), tuple(size))
        if self._map_key == key:
            return self._map
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_raw, self.D, np.eye(3), np.asarray(K_dso, dtype=np.float64),
            (int(size[0]), int(size[1])), cv2.CV_16SC2)
        self._map_key = key
        self._map = (m1, m2)
        return self._map

    def lookup(self, stamp, K_dso, size):
        """Colour image (h,w,3) bgr8 aligned to the keyframe, or None.

        None means no raw frame close enough in time - the caller should fall
        back to the grayscale keyframe rather than colour points with garbage.
        """
        with self._lock:
            if not self._buf:
                self.n_miss += 1
                return None
            k = min(self._buf, key=lambda x: abs(x - stamp))
            if abs(k - stamp) > self.max_time_diff:
                self.n_miss += 1
                return None
            raw = self._buf[k]

        if (raw.shape[1], raw.shape[0]) != self.raw_size:
            # the calibration is for the full-res stream; scale K to match
            # whatever actually arrived rather than silently misaligning
            sx = raw.shape[1] / float(self.raw_size[0])
            sy = raw.shape[0] / float(self.raw_size[1])
            raw = cv2.resize(raw, self.raw_size, interpolation=cv2.INTER_AREA) \
                if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6 else raw

        m1, m2 = self._maps(K_dso, size)
        self.n_hit += 1
        return cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)
