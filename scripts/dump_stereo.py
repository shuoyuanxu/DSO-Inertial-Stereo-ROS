#!/usr/bin/env python3
import rospy, os
import numpy as np
from nav_msgs.msg import Path

OUT = os.path.dirname(os.path.abspath(__file__))

def save(m):
    a = np.array([[p.header.stamp.to_sec(), p.pose.position.x, p.pose.position.y, p.pose.position.z]
                  for p in m.poses])
    np.save(os.path.join(OUT, "p_stereo.npy"), a)

rospy.init_node("stereo_dump")
rospy.Subscriber("/vi_dso/path", Path, save)
rospy.spin()
