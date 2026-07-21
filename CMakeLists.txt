cmake_minimum_required(VERSION 3.5)
project(polytunnel_vio)

# ---------------------------------------------------------------------------
# One ROS package, four build targets:
#   dso_core            static lib  - the (patched) direct Stereo/VI-DSO core
#   vi_dso_live         node        - live 4-mode wrapper (links dso_core)
#   dso_imu_graph_node  node        - GTSAM factor-graph fusion (links gtsam)
#   dense_mapping_node  node        - fuses dense depth into a global cloud (PCL)
#
# The dense reconstruction stage also ships:
#   msg/               - SlidingWindowsMsg (DSO -> MVS), DepthMsg (MVS -> mapper)
#   include/MVS/       - the MVS engine, alongside the DSO core in include/
#   models/            - UniMVSNet pretrained checkpoints
#   scripts/dense_depth_node.py - the depth node (UniMVSNet | stereo backends)
# ---------------------------------------------------------------------------

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy sensor_msgs geometry_msgs nav_msgs std_msgs cv_bridge
  message_generation pcl_ros pcl_conversions)

add_message_files(
  FILES
  SlidingWindowsMsg.msg
  DepthMsg.msg
)
generate_messages(DEPENDENCIES std_msgs sensor_msgs)

# --- system config-mode packages FIRST, before the core's old Find*.cmake
#     modules go on the module path (GTSAM needs the modern Eigen3::Eigen target
#     which the core's FindEigen3.cmake does not provide) ---
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)

# --- now the core's cmake modules for SuiteParse / Glog ---
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(SuiteParse REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(Pangolin 0.2 QUIET)
# pin OpenCV to the system version cv_bridge links against (mixing with a
# /usr/local OpenCV in one process corrupts the heap)
if(EXISTS "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
  set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "" FORCE)
endif()
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs)

add_definitions("-DENABLE_SSE")

# ===========================================================================
# dso_core : the direct-SLAM core library (Release, -march=native for Eigen SIMD)
# ===========================================================================
set(DSO_CORE ${PROJECT_SOURCE_DIR}/include)
set(DSO_CORE_SRC
  ${DSO_CORE}/FullSystem/FullSystem.cpp
  ${DSO_CORE}/FullSystem/FullSystemOptimize.cpp
  ${DSO_CORE}/FullSystem/FullSystemOptPoint.cpp
  ${DSO_CORE}/FullSystem/FullSystemDebugStuff.cpp
  ${DSO_CORE}/FullSystem/FullSystemMarginalize.cpp
  ${DSO_CORE}/FullSystem/Residuals.cpp
  ${DSO_CORE}/FullSystem/CoarseTracker.cpp
  ${DSO_CORE}/FullSystem/CoarseInitializer.cpp
  ${DSO_CORE}/FullSystem/ImmaturePoint.cpp
  ${DSO_CORE}/FullSystem/HessianBlocks.cpp
  ${DSO_CORE}/FullSystem/PixelSelector2.cpp
  ${DSO_CORE}/FullSystem/IMUPreintegrator.cpp
  ${DSO_CORE}/OptimizationBackend/EnergyFunctional.cpp
  ${DSO_CORE}/OptimizationBackend/AccumulatedTopHessian.cpp
  ${DSO_CORE}/OptimizationBackend/AccumulatedSCHessian.cpp
  ${DSO_CORE}/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${DSO_CORE}/util/settings.cpp
  ${DSO_CORE}/util/Undistort.cpp
  ${DSO_CORE}/util/globalCalib.cpp
  ${DSO_CORE}/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
  ${DSO_CORE}/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)

if(Pangolin_FOUND)
  list(APPEND DSO_CORE_SRC
    ${DSO_CORE}/IOWrapper/Pangolin/KeyFrameDisplay.cpp
    ${DSO_CORE}/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
endif()

add_library(dso_core STATIC ${DSO_CORE_SRC})
target_include_directories(dso_core PUBLIC
  ${DSO_CORE}
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  ${EIGEN3_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Glog_INCLUDE_DIRS})
target_compile_options(dso_core PRIVATE -O3 -g -std=c++14 -march=native)
if(Pangolin_FOUND)
  target_include_directories(dso_core PUBLIC ${Pangolin_INCLUDE_DIRS})
  target_compile_definitions(dso_core PUBLIC HAS_PANGOLIN=1)
endif()

# ===========================================================================
# vi_dso_live : live ROS node (same flags as core — shares Eigen ABI)
# ===========================================================================
add_executable(vi_dso_live src/vi_dso_node.cpp)
target_compile_options(vi_dso_live PRIVATE -O3 -g -std=c++14 -march=native)
target_include_directories(vi_dso_live PRIVATE
  ${catkin_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(vi_dso_live
  dso_core ${catkin_LIBRARIES} ${Pangolin_LIBRARIES} ${OpenCV_LIBS}
  ${GLOG_LIBRARY} ${Boost_LIBRARIES} cxsparse)
# our own message headers must be generated before this compiles
add_dependencies(vi_dso_live ${PROJECT_NAME}_generate_messages_cpp)

# ===========================================================================
# dso_imu_graph_node : GTSAM fusion (NO -march=native — GTSAM Eigen ABI; C++17)
# ===========================================================================
add_executable(dso_imu_graph_node src/dso_imu_graph_node.cpp)
target_compile_options(dso_imu_graph_node PRIVATE -O2 -g -std=c++17)
target_include_directories(dso_imu_graph_node PRIVATE
  ${catkin_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dso_imu_graph_node ${catkin_LIBRARIES} gtsam)
add_dependencies(dso_imu_graph_node ${PROJECT_NAME}_generate_messages_cpp)

# ===========================================================================
# dense_mapping_node : fuse dense depth into a global cloud (PCL; no -march=native)
# ===========================================================================
add_executable(dense_mapping_node src/dense_mapping_node.cpp)
target_compile_options(dense_mapping_node PRIVATE -O2 -g -std=c++14)
target_include_directories(dense_mapping_node PRIVATE
  ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dense_mapping_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(dense_mapping_node ${PROJECT_NAME}_generate_messages_cpp)

# ===========================================================================
# python node + MVS engine
# ===========================================================================
catkin_install_python(PROGRAMS
  scripts/dense_depth_node.py
  scripts/capture_dense.py
  scripts/make_dense_gif.py
  scripts/capture_showcase.py
  scripts/make_showcase_gif.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
