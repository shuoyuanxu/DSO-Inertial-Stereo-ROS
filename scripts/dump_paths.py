#!/usr/bin/env python3
import rospy, os
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float64

OUT = os.path.dirname(os.path.abspath(__file__))
scales = []

def save(name, m):
    a = np.array([[p.header.stamp.to_sec(), p.pose.position.x, p.pose.position.y, p.pose.position.z]
                  for p in m.poses])
    np.save(os.path.join(OUT, name), a)

rospy.init_node("path_dump")
rospy.Subscriber("/dso_graph/path", Path, lambda m: save("p_fused.npy", m))
rospy.Subscriber("/dso_graph/dso_path_aligned", Path, lambda m: save("p_dso.npy", m))
rospy.Subscriber("/dso_graph/scale", Float64, lambda m: (scales.append(m.data),
    np.save(os.path.join(OUT, "p_scale.npy"), np.array(scales))))
rospy.spin()
