#!/usr/bin/python

import h5py
import time
import argparse
import numpy as np

import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel

# Transform ros image into numpy array
def get_frame(image_msg, cloud_msg):
    # Get RGB Frame
    try:
        img = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    rgb_frame = np.array(img)
    rgb_frames.append(rgb_frame)

    # Get depth frame
    gen = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))
    depth_frame = np.array(list(gen)).flatten()
    depth_frames.append(depth_frame)

def record_rgb_and_depth(n_secs):
    # Start Ros node and subscribers for rgb and depth data
    rospy.init_node('recorder', anonymous=True)
    s1 = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    s2 = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)

    ts = message_filters.ApproximateTimeSynchronizer([s1, s2], 10, 0.1)
    ts.registerCallback(get_frame)

    # Record for the given amount of time
    start = time.time()
    while time.time() < start + n_secs:
        print "Recording..."
        time.sleep(5)
    print "Finished recording {} frames!".format(len(rgb_frames))
    s1.unregister()
    s2.unregister()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=str, help="The h5 file to store data in")
    parser.add_argument("jobname", type=str, help="The jobname that will be used in the created files")
    parser.add_argument("-n", dest="n_secs", type=int, default=10, help="Number of seconds to record for")
    args = parser.parse_args()

    # Wait a second to start recording
    time.sleep(1)
    print "Recording starting..."

    # Start recording
    rgb_frames = list(); depth_frames = list()
    bridge = CvBridge()
    record_rgb_and_depth(args.n_secs)

    h5_file = h5py.File(args.h5 + '.h5', "a")

    # Save rgb data to hdf5 file
    rgb_frames = np.array(rgb_frames)
    rgb_dataset = h5_file.create_dataset(args.jobname + '_rgb', data=rgb_frames)

    # Save point cloud to hdf5 file
    depth_frames = np.array(depth_frames)
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    depth_dataset = h5_file.create_dataset(args.jobname + '_depth', data=depth_frames, dtype=dt)

    h5_file.close()
