#!/bin/bash
SP=/tmp/claude-1000/-home-shu-catkin-DSO-ws-src/898b3da5-2545-42c7-8230-164fb71df7e1/scratchpad
CAL=/home/shu/catkin_DSO_ws/src/polytunnel_vio/calib/polytunnel
BIN=/home/shu/catkin_DSO_ws/devel/lib/polytunnel_vio
source /opt/ros/noetic/setup.bash
source /home/shu/catkin_DSO_ws/devel/setup.bash
pgrep -x rosmaster >/dev/null || (nohup roscore > "$SP/roscore.log" 2>&1 &)
sleep 3
kill -9 $(ps aux | awk '/[v]i_dso_live|[d]so_imu_graph_node|[d]ump_/{print $2}') 2>/dev/null
sleep 2
rm -f "$SP/traj.npy" "$SP/sg_vi.log" "$SP/sg_graph.log"
cd "$SP"
# stereo (metric) vision node
nohup $BIN/vi_dso_live __name:=vi_dso \
  _calib:=$CAL/cam0.txt _calib_right:=$CAL/cam1.txt _t_stereo:=$CAL/T_C0C1.txt \
  _imu_info:=$CAL/IMU_info.txt _timeshift_cam_imu:=0.009160379134269684 \
  _use_imu:=false _use_stereo:=true _nogui:=true > "$SP/sg_vi.log" 2>&1 &
# graph in METRIC mode (no scale state)
nohup $BIN/dso_imu_graph_node __name:=dso_imu_graph \
  _imu_info:=$CAL/IMU_info.txt _metric_input:=true > "$SP/sg_graph.log" 2>&1 &
# dump the fused graph path
nohup python3 -c "
import rospy, numpy as np
from nav_msgs.msg import Path
def cb(m):
    a=np.array([[p.header.stamp.to_sec(),p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in m.poses])
    np.save('$SP/traj.npy', a)
rospy.init_node('d',anonymous=True); rospy.Subscriber('/dso_graph/path',Path,cb); rospy.spin()" > /dev/null 2>&1 &
sleep 5
rosbag play /home/shu/easy_AprilAdd_tffix.bag -s 20 \
  --topics /forwardLeft/image_raw/compressed /forwardRight/image_raw/compressed /ms/imu/data > "$SP/sg_play.log" 2>&1
sleep 5
cp "$SP/traj.npy" "$SP/traj_stereo_graph.npy" 2>/dev/null
n=$(python3 -c "import numpy as np;print(len(np.load('$SP/traj_stereo_graph.npy')))" 2>/dev/null || echo 0)
echo "stereo_graph: $n poses"
kill -9 $(ps aux | awk '/[v]i_dso_live|[d]so_imu_graph_node|[d]ump_/{print $2}') 2>/dev/null
echo "STEREO GRAPH DONE"
