"""Classical stereo depth backend.

The other half of the benchmark. Where UniMVSNet sweeps planes across several
DSO keyframes (baselines ~0.30 m, but monocular and learned), this uses the
rig's actual calibrated pair (baseline 0.1395 m, metric by construction, no
network). Both publish the same DepthMsg so dense_mapping cannot tell them apart.

The awkward part is frames. DSO hands us an *undistorted pinhole* keyframe with
its own K, produced by DSO's `crop` rectification of the fisheye left camera.
Stereo matching needs a *stereo-rectified* pair, which is a different rotation
of the same camera. So depth is computed in the rectified frame and then
reprojected into the DSO keyframe frame, otherwise every point lands slightly
rotated and the map smears.

Chain per pixel (u,v) of the DSO keyframe:
    d_cam  = K_dso^-1 [u,v,1]        ray in left-camera coords
    d_rect = R1 @ d_cam              same ray in rectified coords
    (u_r,v_r) = pi(K_rect @ d_rect)  where to sample the disparity
    Z_rect = f_rect * B / disp(u_r,v_r)
    X_rect = Z_rect * d_rect / d_rect.z
    depth  = (R1^T @ X_rect).z       back to the DSO keyframe frame
"""
import os
import threading

import numpy as np
import cv2
import rospy

from .result import DepthResult


def read_dso_calib(path):
    """Parse a DSO calibration file. Returns (model, K, D, (w,h))."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    parts = lines[0].split()
    model = parts[0]
    vals = [float(x) for x in parts[1:]]
    w, h = [int(x) for x in lines[1].split()[:2]]
    fx, fy, cx, cy = vals[:4]
    # DSO normalises intrinsics by image size when they are given as fractions
    if fx < 1.0:
        fx *= w; cx *= w; fy *= h; cy *= h
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array(vals[4:8], dtype=np.float64).reshape(4, 1)
    return model, K, D, (w, h)


def read_T(path):
    """Read a 3x4 (or 4x4) extrinsic into a 4x4."""
    rows = []
    with open(path) as f:
        for line in f:
            v = [float(x) for x in line.split()]
            if len(v) == 4:
                rows.append(v)
    T = np.eye(4)
    T[:len(rows), :4] = np.array(rows[:3])
    return T


class StereoBackend(object):
    name = "stereo"

    @classmethod
    def from_params(cls, pkg):
        calib_dir = rospy.get_param(
            "~calib_dir",
            os.path.join(pkg, "calib", "polytunnel"))
        return cls(
            calib_left=rospy.get_param("~calib_left",
                                       os.path.join(calib_dir, "cam0.txt")),
            calib_right=rospy.get_param("~calib_right",
                                        os.path.join(calib_dir, "cam1.txt")),
            t_stereo=rospy.get_param("~t_stereo",
                                     os.path.join(calib_dir, "T_C0C1.txt")),
            left_topic=rospy.get_param("~left_topic",
                                       "/forwardLeft/image_raw/compressed"),
            right_topic=rospy.get_param("~right_topic",
                                        "/forwardRight/image_raw/compressed"),
            rect_w=int(rospy.get_param("~rect_w", 1024)),
            rect_h=int(rospy.get_param("~rect_h", 768)),
            max_depth=float(rospy.get_param("~stereo_max_depth", 12.0)),
            balance=float(rospy.get_param("~rect_balance", 1.0)),
            fov_scale=float(rospy.get_param("~rect_fov_scale", 1.0)),
            min_match_depth=float(rospy.get_param("~min_match_depth", 1.0)),
            block_size=int(rospy.get_param("~block_size", 9)),
            uniqueness_ratio=int(rospy.get_param("~uniqueness_ratio", 5)),
            max_time_diff=float(rospy.get_param("~max_time_diff", 0.02)))

    def __init__(self, calib_left, calib_right, t_stereo,
                 left_topic, right_topic, rect_w=1024, rect_h=768,
                 max_depth=12.0, max_time_diff=0.02, buffer_len=400,
                 balance=1.0, fov_scale=1.0, min_match_depth=1.0,
                 block_size=9, uniqueness_ratio=5):
        _, K1, D1, size1 = read_dso_calib(calib_left)
        _, K2, D2, _ = read_dso_calib(calib_right)

        # T_C0C1 maps points in the RIGHT frame into the LEFT frame.
        # cv2 wants the transform taking LEFT into RIGHT, i.e. its inverse.
        T_l_r = read_T(t_stereo)
        T_r_l = np.linalg.inv(T_l_r)
        R, t = T_r_l[:3, :3], T_r_l[:3, 3]
        self.baseline = float(np.linalg.norm(T_l_r[:3, 3]))

        # rectify at reduced resolution: matching 2048x1536 costs far more than
        # it gains once the result is resampled into a 640x480 keyframe
        self.rect_size = (int(rect_w), int(rect_h))
        s = np.array([[rect_w / float(size1[0]), 0, 0],
                      [0, rect_h / float(size1[1]), 0],
                      [0, 0, 1]], dtype=np.float64)
        K1s, K2s = s @ K1, s @ K2

        # balance=0 crops the rectified image to the all-valid region, which
        # zooms in to ~101 deg - narrower than the 115 deg DSO keyframe we have
        # to resample into. A quarter of the keyframe then samples outside the
        # rectified image and comes back as a hard-zero band on the left/right
        # edges, in every frame. balance=1 keeps the full FOV instead; the outer
        # ring is empty for the matcher but no keyframe pixel is lost.
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1s, D1, K2s, D2, self.rect_size, R, t,
            cv2.CALIB_ZERO_DISPARITY, newImageSize=self.rect_size,
            balance=balance, fov_scale=fov_scale)

        self.R1 = R1
        self.K_rect = P1[:3, :3]
        self.f_rect = float(P1[0, 0])

        self.map1 = cv2.fisheye.initUndistortRectifyMap(
            K1s, D1, R1, P1, self.rect_size, cv2.CV_16SC2)
        self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K2s, D2, R2, P2, self.rect_size, cv2.CV_16SC2)

        self.max_depth = max_depth
        self.max_time_diff = max_time_diff

        # disparity search must cover the depth range we care about:
        # disp = f * B / Z, so the nearest depth sets the widest disparity.
        # It is not free: minDisparity=0 leaves the leftmost num_disp columns
        # of the rectified image unmatchable, so asking for an unnecessarily
        # close min depth buys nothing and costs FOV. 0.5 m is closer than this
        # platform ever needs; 1.0 m halves the dead band.
        num_disp = int(np.ceil(self.f_rect * self.baseline
                               / min_match_depth / 16.0)) * 16
        num_disp = max(64, min(num_disp, 256))
        # blockSize 9 / uniquenessRatio 5 rather than 5 / 10: a polytunnel is
        # mostly untextured plastic and bare soil, and the tighter settings
        # decline to match about half of it. Measured on a rectified pair from
        # the polytunnel bag: 45.3% valid at 5/10, 55.7% at 9/5. The cost is a
        # larger correlation window, so depth edges smooth out a little.
        bs = int(block_size) | 1                      # SGBM requires odd
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=bs,
            P1=8 * 3 * bs ** 2, P2=32 * 3 * bs ** 2,
            disp12MaxDiff=1, uniquenessRatio=int(uniqueness_ratio),
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        hfov = 2.0 * np.degrees(np.arctan(0.5 * rect_w / self.f_rect))
        rospy.loginfo("stereo: baseline %.4f m, rect %dx%d, f_rect %.1f, "
                      "hfov %.1f deg (balance %.2f, fov_scale %.2f), "
                      "numDisparities %d (min depth %.2f m, "
                      "left dead band %d/%d cols)",
                      self.baseline, rect_w, rect_h, self.f_rect, hfov,
                      balance, fov_scale, num_disp,
                      self.f_rect * self.baseline / num_disp, num_disp, rect_w)

        # raw image buffers, keyed by stamp
        self._lock = threading.Lock()
        self._left = {}
        self._right = {}
        self._buffer_len = buffer_len
        self._cache_key = None
        self._cache_map = None

        from sensor_msgs.msg import CompressedImage, Image
        for topic, store in ((left_topic, self._left), (right_topic, self._right)):
            if topic.endswith("/compressed"):
                rospy.Subscriber(topic, CompressedImage,
                                 self._on_img, callback_args=store, queue_size=30)
            else:
                rospy.Subscriber(topic, Image,
                                 self._on_img, callback_args=store, queue_size=30)

    def _on_img(self, msg, store):
        if hasattr(msg, "format"):
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                               cv2.IMREAD_GRAYSCALE)
        else:
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = img[:, :, 0]
        with self._lock:
            store[msg.header.stamp.to_sec()] = img
            if len(store) > self._buffer_len:
                for k in sorted(store)[:len(store) - self._buffer_len]:
                    del store[k]

    def _nearest(self, store, t):
        with self._lock:
            if not store:
                return None, None
            k = min(store, key=lambda x: abs(x - t))
            return store[k], abs(k - t)

    def _dso_to_rect_maps(self, K_dso, shape):
        """Sampling maps from DSO keyframe pixels into the rectified image.

        Cached: K_dso and the image size never change during a run.
        """
        key = (tuple(K_dso.ravel()), shape)
        if self._cache_key == key:
            return self._cache_map

        h, w = shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                           np.arange(h, dtype=np.float64))
        ones = np.ones_like(u)
        pix = np.stack([u, v, ones], axis=-1)                  # (h,w,3)
        d_cam = pix @ np.linalg.inv(K_dso).T                   # rays, z=1
        d_rect = d_cam @ self.R1.T                             # into rectified frame
        p = d_rect @ self.K_rect.T
        z = p[..., 2:3]
        bad = np.abs(z[..., 0]) < 1e-9
        uv = p[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)

        res = (uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
               d_rect, bad)
        self._cache_key = key
        self._cache_map = res
        return res

    def run(self, window, images):
        ref = window.ref_index()
        stamp = window.views[ref].image.header.stamp.to_sec()

        left, dl = self._nearest(self._left, stamp)
        right, dr = self._nearest(self._right, stamp)
        if left is None or right is None:
            raise RuntimeError("no raw stereo images buffered yet")
        if dl > self.max_time_diff or dr > self.max_time_diff:
            raise RuntimeError("no stereo pair within %.3f s of keyframe "
                               "(left %.3f, right %.3f)"
                               % (self.max_time_diff, dl, dr))

        rl = cv2.remap(left, self.map1[0], self.map1[1], cv2.INTER_LINEAR)
        rr = cv2.remap(right, self.map2[0], self.map2[1], cv2.INTER_LINEAR)

        disp = self.matcher.compute(rl, rr).astype(np.float32) / 16.0
        valid = disp > 0.5
        z_rect = np.zeros_like(disp)
        z_rect[valid] = self.f_rect * self.baseline / disp[valid]

        K_dso = window.K(ref)
        h = window.views[ref].image.height
        w = window.views[ref].image.width
        mx, my, d_rect, bad = self._dso_to_rect_maps(K_dso, (h, w))

        z_s = cv2.remap(z_rect, mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v_s = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Z in the rectified frame is along the rectified optical axis; convert
        # to depth along the DSO keyframe's axis by walking the ray back
        scale = np.where(np.abs(d_rect[..., 2]) < 1e-9, 0.0,
                         1.0 / np.where(np.abs(d_rect[..., 2]) < 1e-9, 1.0,
                                        d_rect[..., 2]))
        X_rect = d_rect * (z_s * scale)[..., None]
        X_cam = X_rect @ self.R1                      # R1^T applied on the right
        depth = X_cam[..., 2].astype(np.float32)

        ok = (v_s > 0) & (~bad) & np.isfinite(depth) & (depth > 0) \
            & (depth < self.max_depth)
        depth[~ok] = 0.0

        # SGBM gives no probability; use a geometric proxy so dense_mapping's
        # confidence gate means something comparable. Disparity precision is
        # ~1/16 px, so relative depth error grows with depth: conf ~ 1 - Z/Zmax.
        conf = np.zeros_like(depth)
        conf[ok] = np.clip(1.0 - depth[ok] / self.max_depth, 0.0, 1.0)

        return DepthResult(depth=depth, confidence=conf, ref_index=ref,
                           K=K_dso, backend=self.name)
