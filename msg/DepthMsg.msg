# Dense depth for one reference keyframe, published by dense_depth.
# Backend-agnostic: both the UniMVSNet and the stereo backend emit this,
# so dense_mapping does not know which produced it.

sensor_msgs/Image image      # reference view, bgr8, same size as depth
float64[16] camToWorld       # rigid camera->world of the reference view, row-major 4x4
float64[4]  Intrinsics       # fx, fy, cx, cy in pixels of `depth`

sensor_msgs/Image depth      # 32FC1, metres. <=0 or non-finite means no estimate.
sensor_msgs/Image confidence # 32FC1, 0..1. Higher is better; dense_mapping thresholds on it.

string backend               # "unimvsnet" or "stereo", for benchmarking/provenance
