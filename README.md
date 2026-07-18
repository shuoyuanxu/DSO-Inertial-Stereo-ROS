# Polytunnel Direct Visual(-Inertial) Odometry

A ROS1 (noetic) direct sparse odometry suite for agricultural robots, developed and
field-validated on a heavy row-crop tractor in a polytunnel (2048×1536 fisheye cameras
@ 10 Hz, 200 Hz IMU, near-constant-velocity motion — one of the hardest regimes for
visual-inertial scale estimation).

Built on [VI-Stereo-DSO](https://github.com/RonaldSun/VI-Stereo-DSO) (included here as a
patched fork — see [Upstream fixes](#upstream-fixes-in-vi-stereo-dso)), with a
GTSAM-based loosely-coupled fusion layer inspired by the design principles of
[OKVIS2](https://github.com/smartroboticslab/okvis2).

## Repository layout

A single ROS1 (catkin) package. Three build targets are produced from it: the
DSO core library, the live odometry node, and the fusion node.

| Path | What it is |
|---|---|
| `include/` | the direct Stereo/VI-DSO **core** (patched third-party DSO, GPLv3) — headers + sources, built into the internal `dso_core` static library. All our crash/scale fixes live here |
| `src/vi_dso_node.cpp` | **(ours)** live ROS node with four modes (`mono`, `mono_imu`, `stereo`, `stereo_imu`), full RViz visualization, OKVIS-style init relaxations → executable `vi_dso_live` |
| `src/dso_imu_graph_node.cpp` | **(ours)** GTSAM iSAM2 factor graph fusing mono DSO odometry with preintegrated IMU factors → executable `dso_imu_graph_node` |
| `launch/` | `vi_dso.launch` (all four modes via `mode:=…`), `dso_imu_graph.launch` |
| `rviz/`, `calib/` | RViz layouts and calibration files (incl. `calib/polytunnel/source_calibration.yaml`, the original Kalibr calibration everything was derived from) |
| `scripts/` | **(ours)** trajectory recorders + `eval_ate.py` (SE3 + Sim3 Umeyama ATE) |
| `thirdparty/` | Sophus + sse2neon headers the core needs |
| `cmake/` | `Find*` modules for SuiteSparse / Glog |

`vi_dso_live` links `dso_core`; `dso_imu_graph_node` links GTSAM. The two are
independent executables in one package — build once, run either.

## Modes

One complete, standalone launch file per mode (each starts the node + RViz):

```bash
roslaunch polytunnel_vio stereo.launch        # RECOMMENDED — metric, range-gated stereo
roslaunch polytunnel_vio mono.launch          # pure vision, best shape, scale-free
roslaunch polytunnel_vio mono_graph.launch    # mono + IMU loose fusion (gravity-aligned metric)
roslaunch polytunnel_vio stereo_graph.launch  # stereo + IMU loose fusion (no scale state)
roslaunch polytunnel_vio mono_imu.launch      # tightly-coupled VI  — marginal, not recommended
roslaunch polytunnel_vio stereo_imu.launch    # tightly-coupled stereo-VI — not recommended

# then, in another terminal:
rosbag play easy_AprilAdd_tffix.bag
```

(`vi_dso.launch` is the shared base the per-mode files include — you can also call it
directly with `mode:=…` and `rviz_config:=…` if you want.)

The launch file auto-starts RViz (`rviz:=false` to disable) showing the raw image, the
keyframe image with selected points (colored by inverse depth), the sparse map, odometry
and path. In the graph configuration the map is assembled from per-keyframe clouds using
the *fused* (IMU-corrected, metric, gravity-aligned) poses.

## Results (polytunnel sequence, tag-map ground truth, ~327 s)

| System | SE3 ATE (metric) | Sim3 ATE (shape) | scale err |
|---|---|---|---|
| **ours: stereo (range-gated)** | **0.88 m** | **0.47 m** | **1.04** |
| OpenVINS (mono-inertial) | 0.63 m | 0.61 m | 0.992 |
| ORB-SLAM3 (mono-inertial) | 0.86 m | 0.80 m | 1.018 |
| ours: mono (vision only) | — (scale-free) | 0.19 m | — |
| ours: mono + IMU graph | (scale run-dependent) | 0.31 m | unreliable |
| VINS-Fusion (stereo-inertial) | 3.19 m | 3.06 m | 1.050 |
| ours: stereo (ungated) | 8.07 m | 6.86 m | 1.29 |
| VINS-Fusion (mono-inertial) | 15.9 m | (diverged) | — |

**The winner is range-gated stereo, no IMU needed** (SE3 0.88 m, scale within 4 %),
competitive with the feature-based leaders. The key fix: a 14 cm baseline cannot
triangulate far structure, and DSO was baking that garbage depth into the map. Dropping
static-stereo points past ~12 m (`stereo_max_depth`) took stereo from Sim3 6.86 m → 0.47 m
— a ~15× gain. Once stereo is clean, adding IMU fusion *hurts* (it only ever compensated
for the broken depth), so the recommended metric config is plain `mode:=stereo`.

Other notes: **mono alone has the best raw shape** (Sim3 0.19 m) but no metric scale;
the **mono+IMU graph** gives good shape (0.31 m) but IMU-only scale is unreliable on
constant-velocity motion; the tightly-coupled `*_imu` modes are marginal on this data
(scale is weakly observable) and are not recommended.

## Build

Clone this repo into `<catkin_ws>/src/` and build the whole thing in one shot —
the core library, both nodes, everything:

```bash
cd <catkin_ws>
catkin_make -DCATKIN_WHITELIST_PACKAGES="polytunnel_vio"
```

The CMake handles the fiddly bits automatically: the core + `vi_dso_live` build
with `-march=native` (Eigen SIMD), while `dso_imu_graph_node` builds without it and
in C++17 (GTSAM's Eigen ABI), and OpenCV is pinned to the system version cv_bridge
links against (mixing two OpenCVs in one process corrupts the heap).

Dependencies: ROS noetic, Pangolin, GTSAM ≥ 4.2 (`CombinedImuFactor`), OpenCV (system),
Eigen3, Boost, SuiteSparse, glog.

## Calibration inputs

- `calib/<rig>/cam0.txt`, `cam1.txt` — DSO calibration format
  (EquiDistant/RadTan intrinsics, input size, `crop`, output size)
- `IMU_info.txt` — 3×4 `T_imu_cam` rows, a skipped line, then gyro/accel noise
  densities and random walks (Kalibr conventions)
- `T_C0C1.txt` — 3×4 left←right stereo extrinsic

## Key parameters (learned the hard way)

| Param | Default | Why |
|---|---|---|
| `imu_weight` | 1.0–2.0 | upstream's 6.0 was drone-tuned; inflated IMU residuals broke the initializer's RMSE gates |
| `init_slack` | 3.0 | relaxes the early keyframe photometric gates (OKVIS-style: accept early, refine online) |
| `scale_reset_low/high` | 0.02 / 50 | upstream reset the whole system when scale left [0.1, 10] — fatal on low-excitation platforms |
| `carry_state` | true | preserves gyro/accel bias and scale estimates across DSO re-initializations |
| `acc_sigma_inflation` (graph) | 5 | bench-calibrated noise densities are far too optimistic on a vibrating vehicle (OKVIS2 configs ship 5–20× inflated values) |
| `scale_rw_sigma` | 0.002 | scale random walk; larger values let scale chase IMU noise through unobservable stretches |
| `odom_huber` | 1.345 | robust loss on odometry factors; disabling it measurably hurts |

## Upstream fixes in VI-Stereo-DSO

This tree fixes the following upstream bugs (all found on real data):

1. **Heap buffer overflow in `CoarseTracker::trackNewestCoarse`** — `imu_track_w` was
   sized `coarsestLvl` but 5 elements are always written; with image geometries that
   yield 4 pyramid levels this smashed heap metadata **every frame**. (EuRoC's geometry
   hid it in allocator slack — found with AddressSanitizer.)
2. **Async mapping queue desync** — `deliverTrackedFrame` never pushed `fh_right` to
   `unmappedTrackedFrames_right` (pop on empty deque = UB), and the catch-up path popped
   only the left queue.
3. **Unguarded IMU reads** in `EnergyFunctional::getIMUHessian`, the coarse tracker, and
   `initFirstFrame_imu` — vision-only operation crashed; now guarded (enables `mono` /
   `stereo` modes without IMU data).
4. **Scale divergence handling** — scale steps are clamped (with configurable bounds)
   instead of triggering full resets; non-finite optimizer states trigger clean resets
   instead of `Sophus::ScaleNotPositive` aborts; stereo mode's hardcoded assumption that
   the baseline pins scale to [0.6, 2] is now configurable (false for short-baseline
   rigs observing far structure).
5. GUI event-loop calls (`cv::waitKey`) no longer run when display is disabled.

## Known issues / roadmap

- **Graph scale bias (~25 %)**: keyframe-level assembly loses scale information relative
  to full-rate tight coupling (OpenVINS recovers scale 0.99 from the same IMU). Raw
  per-turn IMU ΔV forensics show the sensor stream is unbiased — first fix candidate is
  denser keyframes during turns (0.5 s spans ~17° at turn rate).
- **stereo_imu scale correction**: the tightly-coupled optimizer destabilizes on large
  scale corrections (NaN → now a clean reset instead of a crash, but convergence through
  the correction is unvalidated). `stereo_weight` reduction and the graph-based route
  are workarounds.
- Intermittent crash in the *async* mapping mode near resets (sync mode, the default, is
  unaffected).
- Sync+single-thread mode runs ~0.26× real time in `mono_imu`; use
  `multithreading:=true linearize_operation:=false` (real-time) or slow the bag.

## Acknowledgements & citations

This work builds directly on:

- **DSO** — J. Engel, V. Koltun, D. Cremers, *Direct Sparse Odometry*, IEEE TPAMI 2018.
  [github.com/JakobEngel/dso](https://github.com/JakobEngel/dso)
- **Stereo DSO** — R. Wang, M. Schwörer, D. Cremers, *Stereo DSO: Large-Scale Direct
  Sparse Visual Odometry with Stereo Cameras*, ICCV 2017.
- **VI-DSO** — L. von Stumberg, V. Usenko, D. Cremers, *Direct Sparse Visual-Inertial
  Odometry using Dynamic Marginalization*, ICRA 2018.
- **VI-Stereo-DSO** — R. Sun's community implementation combining the above:
  [github.com/RonaldSun/VI-Stereo-DSO](https://github.com/RonaldSun/VI-Stereo-DSO)
  (included here in patched form, GPLv3)
- **dso_ros** — J. Engel's ROS wrapper for DSO, the pattern our live wrapper follows:
  [github.com/JakobEngel/dso_ros](https://github.com/JakobEngel/dso_ros)
- **GTSAM** — F. Dellaert et al., Georgia Tech Smoothing and Mapping library; IMU
  preintegration after C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza, *On-Manifold
  Preintegration for Real-Time Visual-Inertial Odometry*, IEEE T-RO 2017.
  [gtsam.org](https://gtsam.org)
- **OKVIS2 / OKVIS2-X** — S. Leutenegger et al., whose estimator design (immediate
  initialization without excitation gates, information-preserving marginalization,
  physically-derived IMU weighting with config-level noise inflation, persistent bias
  priors) guided the fusion layer and the initialization relaxations:
  [github.com/smartroboticslab/okvis2](https://github.com/smartroboticslab/okvis2)

Baselines referenced in the results table: OpenVINS (Geneva et al.), ORB-SLAM3 (Campos
et al.), VINS-Fusion (Qin et al.).

## License

GPLv3 — this repository contains and derives from DSO-family code, which is GPLv3.
The `dso_imu_graph` package and `tools/` are also released under GPLv3 for consistency.
