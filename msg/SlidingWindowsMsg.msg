# One keyframe of a DSO sliding window, published by vi_dso_live.
#
# All keyframes belonging to the same window share msg_id; a subscriber
# buffers messages until it holds window_size of them, then runs MVS.
# msg_id increments every time the window is published, so a subscriber
# that sees a new id can drop a partial buffer without ambiguity.

uint64 msg_id           # window identity; constant across one window
uint8  window_size      # number of keyframes in this window
uint8  index            # position of this keyframe within the window, 0-based

sensor_msgs/Image image # undistorted keyframe DSO actually optimised, bgr8

# Rigid camera->world pose, row-major 4x4, metric (T_WD scale already applied
# to the translation; the rotation block is orthonormal). Same world frame as
# vi_dso/cloud and vi_dso/pose.
float64[16] camToWorld

# Pinhole intrinsics of the undistorted image, level 0: fx, fy, cx, cy.
# Already in pixels of `image` - do NOT rescale.
float64[4] Intrinsics

# Plane-sweep depth range for this window, in metres, derived from the
# percentiles of DSO's own sparse inverse depths. In mono these are in DSO
# units scaled by T_WD like the pose, so they stay consistent with camToWorld.
float32 depth_min
float32 depth_max
