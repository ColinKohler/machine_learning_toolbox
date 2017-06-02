#!/usr/bin/python

import rospy
import StringIO
import shutil
import time
from sensor_msgs.msg import CameraInfo

def get_camera_info(cam_info_msg):
    buf = StringIO.StringIO()
    cam_info_msg.serialize(buf)

    with open('camera_info.txt', 'w') as fd:
        buf.seek(0)
        shutil.copyfileobj(buf, fd)

if __name__ == '__main__':
    rospy.init_node('get_cam_info', anonymous=True)
    s = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, get_camera_info)

    start = time.time()
    while time.time() < start + 2:
        time.sleep(1)
    s.unregister()
