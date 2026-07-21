
# Direct Sparse(-Inertial) Odometry

![stereo_showcase](https://github.com/user-attachments/assets/8d2158d1-db2c-4ce5-8e1e-cf124640d768)

Vibe coded DSO ros wrapper with: mono, mono inertial, stereo, stereo inertial, GTSAM style VI fusion,
and dense reconstruction (UniMVSNet)

Built on [VI-Stereo-DSO](https://github.com/RonaldSun/VI-Stereo-DSO) (included here as a
patched fork — see [Upstream fixes](#upstream-fixes-in-vi-stereo-dso)), with a
GTSAM-based loosely-coupled fusion layer inspired by the design principles of
[OKVIS2](https://github.com/smartroboticslab/okvis2).

## Repository layout

A single ROS1 (catkin) package. Four build targets are produced from it: the
DSO core library, the live odometry node, the fusion node, and the dense mapper.

| Path | What it is |
|---|---|
| `include/` | the direct Stereo/VI-DSO **core** 
| `include/MVS/` | **(ours)** the MVS engine — UniMVSNet (upstream `networks/` unmodified) + stereo backend |
| `msg/` | **(ours)** `SlidingWindowsMsg` (DSO → MVS), `DepthMsg` (MVS → mapper) |
| `models/` | UniMVSNet pretrained checkpoints (DTU + BlendedMVS) |
| `src/vi_dso_node.cpp` | **(ours)** live ROS node with four modes (`mono`, `mono_imu`, `stereo`, `stereo_imu`), full RViz visualization, OKVIS-style init relaxations → executable `vi_dso_live` |
| `src/dso_imu_graph_node.cpp` | **(ours)** GTSAM iSAM2 factor graph fusing mono DSO odometry with preintegrated IMU factors → executable `dso_imu_graph_node` |
| `src/dense_mapping_node.cpp` | **(ours)** fuses dense depth into a global voxel-filtered cloud → executable `dense_mapping_node` |
| `scripts/dense_depth_node.py` | **(ours)** dense depth node, backend selectable (`unimvsnet` \| `stereo`) |
| `scripts/` | **(ours)** trajectory recorders + `eval_ate.py` (SE3 + Sim3 Umeyama ATE) |
| `thirdparty/` | Sophus + sse2neon headers the core needs |

`vi_dso_live` links `dso_core`; `dso_imu_graph_node` links GTSAM; `dense_mapping_node`
links PCL. All independent executables in one package — build once, run any.

## Modes

One standalone launch file per mode (each starts the node + RViz):

```bash
roslaunch polytunnel_vio stereo.launch        
roslaunch polytunnel_vio mono.launch          
roslaunch polytunnel_vio mono_graph.launch    # mono + IMU loose fusion
roslaunch polytunnel_vio stereo_graph.launch  # stereo + IMU loose fusion
roslaunch polytunnel_vio mono_imu.launch      # tightly-coupled VI
roslaunch polytunnel_vio stereo_imu.launch    # tightly-coupled stereo-VI

# then, in another terminal:
rosbag play your.bag
```

## Results (polytunnel sequence, tag-map ground truth, ~343 s)

| System | Modality | SE3 ATE | Sim3 ATE | scale | coverage |
|---|---|---|---|---|---|
| **OURS mono** | mono | — scale-free | **0.23 m** | — | 66 % \* |
| ORB-SLAM3 | mono | — scale-free | 0.31 m | — | 78 % |
| DSO (upstream) | mono | — scale-free | 0.31 m | — | 98 % |
| ORB-SLAM3 | stereo | 0.64 m | 0.52 m | 1.03 | 25 % |
| OpenVINS | mono-inertial | **0.63 m** | 0.62 m | 0.99 | 100 % |
| ORB-SLAM3 | mono-inertial | 0.86 m | 0.80 m | 1.02 | 99 % |
| **OURS stereo (range-gated)** | stereo | **1.22 m** | **0.92 m** | **1.04** | **95 %** |
| OURS stereo-inertial | stereo-inertial | 1.30 m | 0.97 m | 1.84 | 1 % |
| VINS-Fusion | stereo-inertial | 3.19 m | 3.06 m | 1.05 | 100 % |
| VINS-Fusion | mono-inertial | 15.9 m | 15.8 m | 0.89 | 100 % |
| OURS mono-inertial | mono-inertial | 22.7 m | 21.3 m | 0.52 | 6 % |
| OpenVINS | stereo-inertial | *diverged* | *diverged* | 0.00 | 100 % |
| VINS-Fusion | stereo | *diverged* | *diverged* | 0.00 | 100 % |

## Dense reconstruction

Dense mapping on the same odometry, after
[DSO + MVSNet](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet).
`vi_dso_live` publishes each keyframe of its sliding window (image, rigid metric
pose, intrinsics, depth range from its own sparse points); UniMVSNet turns each
window into depth + confidence; the mapper fuses them into a global cloud.

![dense showcase](readme_assets/dense_showcase.gif)

```bash
roslaunch polytunnel_vio dense_mvsnet.launch   # UniMVSNet multi-view stereo
roslaunch polytunnel_vio dense_stereo.launch   # classical stereo baseline
roslaunch polytunnel_vio dense_mono.launch     # MVS on mono DSO (shape only)
```

| Backend | Rate | VRAM | Baseline | Valid px |
|---|---|---|---|---|
| **unimvsnet** | 0.19 s/window (5.2 Hz) | 1.1 GB | ~0.30 m (DSO window) | 100 % |
| stereo | 0.09 s/window (11 Hz) | — | 0.1395 m (rig pair) | ~39 % |

Both emit the same `DepthMsg`, so switching is one launch argument. At 12 m a
14 cm baseline gives only 4.9 px of disparity, which SGBM rejects — the same
short-baseline limit `stereo_max_depth` works around in the odometry. The MVS
window baseline is roughly twice that.

## Dependencies: 

ROS noetic, Pangolin, GTSAM ≥ 4.2 (`CombinedImuFactor`), OpenCV (system),
Eigen3, Boost, SuiteSparse, glog.

Dense stage also: `pcl_ros`, `pcl_conversions`, `message_generation`, and PyTorch
with CUDA in the system python3 (verified on torch 2.1.0+cu121). `stereo` needs no GPU.

## Calibration inputs

- `calib/<rig>/cam0.txt`, `cam1.txt` — DSO calibration format
  (EquiDistant/RadTan intrinsics, input size, `crop`, output size)
- `IMU_info.txt` — 3×4 `T_imu_cam` rows, a skipped line, then gyro/accel noise
  densities and random walks (Kalibr conventions)
- `T_C0C1.txt` — 3×4 left←right stereo extrinsic

## Key parameters 

| Param | Default | Why |
|---|---|---|
| `imu_weight` | 1.0–2.0 | upstream's 6.0 was drone-tuned; inflated IMU residuals broke the initializer's RMSE gates |
| `init_slack` | 3.0 | relaxes the early keyframe photometric gates (OKVIS-style: accept early, refine online) |
| `scale_reset_low/high` | 0.02 / 50 | upstream reset the whole system when scale left [0.1, 10] — fatal on low-excitation platforms |
| `carry_state` | true | preserves gyro/accel bias and scale estimates across DSO re-initializations |
| `acc_sigma_inflation` (graph) | 5 | bench-calibrated noise densities are far too optimistic on a vibrating vehicle (OKVIS2 configs ship 5–20× inflated values) |
| `scale_rw_sigma` | 0.002 | scale random walk; larger values let scale chase IMU noise through unobservable stretches |
| `odom_huber` | 1.345 | robust loss on odometry factors; disabling it measurably hurts |

## This work builds directly on:

- **DSO**
  [github.com/JakobEngel/dso](https://github.com/JakobEngel/dso)
- **VI-Stereo-DSO**
  [github.com/RonaldSun/VI-Stereo-DSO](https://github.com/RonaldSun/VI-Stereo-DSO)
  (included here in patched form, GPLv3)
- **dso_ros**
  [github.com/JakobEngel/dso_ros](https://github.com/JakobEngel/dso_ros)
- **GTSAM** 
  [gtsam.org](https://gtsam.org)
- **OKVIS2 / OKVIS2-X**
  [github.com/smartroboticslab/okvis2](https://github.com/smartroboticslab/okvis2)
- **UniMVSNet** — Peng et al., CVPR 2022
  [github.com/prstrive/UniMVSNet](https://github.com/prstrive/UniMVSNet)
- **DSO + MVSNet**
  [Real-time Pose Estimation and Dense Reconstruction](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet)

## License

GPLv3 — this repository contains and derives from DSO-family code, which is GPLv3.
The `dso_imu_graph` package and `tools/` are also released under GPLv3 for consistency.
