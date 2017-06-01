#!/usr/bin/python

import os
import json
import cv2
import h5py
import pcl
import math
import argparse
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import namedtuple
from sklearn.cluster import MeanShift, estimate_bandwidth

from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

BBox = namedtuple('BBox', ['min', 'max'])

def main(jobname, depth_frames, bgr_frames, frames_range):
    # Construct Camera model to convert from 3d points to 2d points
    cam_info = CameraInfo()
    with open('hdf5/camera_info.txt', 'r') as fd:
        cam_data = fd.read()
    cam_info.deserialize(cam_data)
    cam_model = PinholeCameraModel()
    cam_model.fromCameraInfo(cam_info)

    # Loop through desired frames
    bbox_areas = list(); percepts = list()
    for i in frames_range:
        print "Segmenting Frame {}...".format(i)

        # Get frames and convert BGR frame to rgb
        depth_frame = depth_frames[i]
        bgr_frame = bgr_frames[i]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Get point cloud out of depth frame
        num_points = depth_frame.shape[0] / 3
        cloud = depth_frame.reshape((num_points, 3))
        cloud = pcl.PointCloud(cloud.astype(np.float32))

        # Segment bag and convert 3d points to 2d
        bag_cloud = segment_bag_from_cloud(cloud)
        bbox = get_cloud_limits(bag_cloud, cam_model)
        valid, bbox_areas = validate_bbox(bbox, bbox_areas)
        if not valid:
            print "Rejecting frame due to bad bounding box..."
            continue

        # Augment data
        percepts = augment_percept(jobname, rgb_frame, i, bbox, percepts)

    # Save percepts
    with open(jobname + '_percepts.json', 'w') as p_file:
        json.dump(percepts, p_file)

# Segment bag from point cloud
def segment_bag_from_cloud(cloud):
    # Remove points outside of workspace
    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name('z')
    fil.set_filter_limits(0, 2.35)
    cloud_filtered = fil.filter()

    # Remove floor plane
    seg = cloud_filtered.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.06)
    indices, model = seg.segment()

    cloud_bag = cloud_filtered.extract(indices, negative=True)

    # Remove outliers from bag point cloud
    fil = cloud_bag.make_statistical_outlier_filter()
    fil.set_mean_k(100)
    fil.set_std_dev_mul_thresh(0.1)
    cloud_bag_fil = fil.filter()

    # Cluster points and take the largest cluster as the bag
    np_bag_fil = np.asarray(cloud_bag_fil)
    bandwidth = estimate_bandwidth(np_bag_fil, quantile=0.65, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-2)
    ms.fit(np_bag_fil)
    labels = ms.labels_
    cluster_sizes = np.bincount(labels)
    bag_cluster_label = np.argmax(cluster_sizes)
    bag_cluster = np_bag_fil[np.where(labels == bag_cluster_label)]

    bag_cloud = pcl.PointCloud(bag_cluster.astype(np.float32))
    return bag_cloud

# Augment percepts for use in training
def augment_percept(jobname, rgb_frame, img_num, bbox, percepts, n_augments=10):
    # Crop out 80 pixels on each side and resize the image to 256x256
    rgb_frame, bbox = crop_image_lr(rgb_frame, bbox, 80)
    rgb_frame, bbox = resize_image(rgb_frame, bbox)
    frame_size = rgb_frame.shape[0]

    # Apply random perturbations to the image
    for i in range(n_augments):
        # Flip image half the time
        if np.random.random_sample() >= 0.5:
            frame, bbox_tmp = flip_image(rgb_frame, bbox)
            bbox_tmp = BBox(create_h_point(bbox_tmp.min), create_h_point(bbox_tmp.max))
        else:
            frame = rgb_frame
            bbox_tmp = bbox

        # Get random amount of rotation and translation
        rot = np.random.uniform(-12, 12)
        trans_x = np.random.uniform(-5, 5)
        trans_y = np.random.uniform(-5, 5)

        # Create rotation matrix at the center of image
        center = (frame_size/2, frame_size/2)
        H = cv2.getRotationMatrix2D(center, rot, 1.0)
        H[0,-1] += trans_x
        H[1,-1] += trans_y

        # Transform the image and the bounding box
        frame = cv2.warpAffine(frame, H, (frame_size, frame_size))
        bbox_tmp = BBox(create_h_point(bbox_tmp.min), create_h_point(bbox_tmp.max))
        min_bbox_tmp = H.dot(bbox_tmp.min)
        max_bbox_tmp = H.dot(bbox_tmp.max)
        min_bbox_tmp = [round(min_bbox_tmp[0]), round(min_bbox_tmp[1])]
        max_bbox_tmp = [round(max_bbox_tmp[0]), round(max_bbox_tmp[1])]
        bbox_tmp = BBox(min_bbox_tmp, max_bbox_tmp)

        # Add random gaussian noise
        frame = add_noise_to_image(frame)

        # Generate percept
        if (i % 5) < 3:
            dset = 'train'
        elif (i % 5) == 3:
            dset = 'test'
        else:
            dset = 'validate'
        filepath = kDataPath + dset + '{}_{}_{}.jpg'.format(jobname, img_num, i)

        percept = create_percept(filepath, dset, frame, bbox_tmp)
        percepts.append(percept)
        plot_frame_with_box(frame, bbox_tmp, filepath)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(filepath, frame)

    return percepts

# Create the percept annotations for the given image
def create_percept(filepath, dset, frame, bbox):
    percept = dict()
    percept['source'] = 'file://' + filepath
    percept['locator'] = 'file://' + filepath
    percept['format'] = 'image/jpg'
    percept['x_size'] = frame.shape[0]
    percept['y_size'] = frame.shape[1]
    percept['tags'] = ["Warthog", "trash.bag", dset]

    bbox = [[bbox.min[0], bbox.min[1]],
            [bbox.max[0], bbox.min[1]],
            [bbox.max[0], bbox.max[1]],
            [bbox.min[0], bbox.max[1]]]
    percept['annotations'] = [{'domain' : 'warthog',
                               'confidence': 1,
                               'boundary' : bbox,
                               'annotation_tags' : ['auto.segmented']}]
    return percept

# Get the bounding box from the point cloud
def get_cloud_limits(cloud, cam_model):
    n_cloud = np.asarray(cloud)

    min_x = np.min(n_cloud[:,0])
    min_y = np.min(n_cloud[:,1])
    min_z = np.min(n_cloud[:,2])
    min_bbox = [min_x, min_y, min_z]

    max_x = np.max(n_cloud[:,0])
    max_y = np.max(n_cloud[:,1])
    max_z = np.max(n_cloud[:,2])
    max_bbox = [max_x, max_y, max_z]

    min_bbox = cam_model.project3dToPixel(min_bbox)
    max_bbox = cam_model.project3dToPixel(max_bbox)

    bbox = BBox(min_bbox, max_bbox)
    bbox = pad_bbox(bbox)

    return bbox

# Pad bounding box
def pad_bbox(bbox, pad=25):
    min_bbox = [bbox.min[0], bbox.min[1] - pad]
    max_bbox = [bbox.max[0] + pad, bbox.max[1] + pad]
    return BBox(min_bbox, max_bbox)

# Check if bounding box is a reasonable size
def validate_bbox(bbox, bbox_areas):
    bbox_area = (bbox.max[0] - bbox.min[0]) * (bbox.max[1] - bbox.min[1])
    mean_bbox_area = np.mean(np.array(bbox_areas)) if bbox_areas else 0

    valid = True
    if len(bbox_areas) < 10:
        bbox_areas.append(bbox_area)
    elif bbox_area < 2 *  mean_bbox_area and bbox_area > 0.25 * mean_bbox_area:
        bbox_areas.append(bbox_area)
    else:
        valid = False

    return valid, bbox_areas

# Crop out pixels on the sides to make image square
def crop_image_lr(rgb_frame, bbox, pixels):
    rgb_frame = rgb_frame[:,pixels:-pixels,:]
    frame_shape = rgb_frame.shape

    # Adjust bounding box
    min_bbox = (max(bbox.min[0] - pixels, 0), bbox.min[1])
    max_bbox = (min(bbox.max[0] - pixels, frame_shape[1]), bbox.max[1])

    return rgb_frame, BBox(min_bbox, max_bbox)

# Resize image to be 256x256 and adjust bounding box
def resize_image(rgb_frame, bbox):
    rgb_frame = cv2.resize(rgb_frame, (256, 256))
    min_bbox = [bbox.min[0] / 2, bbox.min[1] / 2]
    max_bbox = [bbox.max[0] / 2, bbox.max[1] / 2]

    return rgb_frame, BBox(min_bbox, max_bbox)

# Flip (left/right) a image and its bounding box
def flip_image(rgb_frame, bbox):
    rgb_frame = cv2.flip(rgb_frame, 1)
    img_size = rgb_frame.shape[0]
    min_to_side = [img_size - bbox.min[0], bbox.min[1]]
    max_to_side = [img_size - bbox.max[0], bbox.max[1]]

    return rgb_frame, BBox(min_to_side, max_to_side)

# Add random gausian noise to image
def add_noise_to_image(rgb_frame):
    noise = np.zeros(rgb_frame.shape, np.uint8)
    cv2.randn(noise, np.zeros(3), np.ones(3)*255*0.1)
    rgb_frame = cv2.add(rgb_frame, noise, dtype=cv2.CV_8UC3)

    return rgb_frame

# Turn a 2d point into a homogeneous 2d point
def create_h_point(p):
    return [p[0], p[1], 1.0]

# Save frame with bounding box displayed
def plot_frame_with_box(rgb_frame, bbox, filepath):
    fig, ax = plt.subplots(1)
    ax.imshow(rgb_frame)
    diff_x = bbox.max[0] - bbox.min[0]
    diff_y = bbox.max[1] - bbox.min[1]
    rect = patches.Rectangle(bbox.min, diff_x, diff_y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.savefig(filepath)
    plt.close(fig)

if __name__ == '__main__':
    # Parser junk
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=str, help="The h5 file containing the point cloud")
    parser.add_argument("jobname", type=str, help="The jobname to run the segmentation on")
    parser.add_argument("datapath", type=str, help="The path to the data to be utilized")
    parser.add_argument("-n", dest='n', type=int, default=-1, help="The desired rgb frame number, or -1 (default) for all")
    args = parser.parse_args()

    #kDataPath = '/home/ur5/external/warthog/data/'
    kDataPath = args.datapath

    # Get desired frame from number
    h5_file = h5py.File(kDataPath + args.h5 + '.h5', 'r')
    depth_frames = h5_file[args.jobname + '_depth']
    bgr_frames = h5_file[args.jobname + '_rgb']

    # Determine range from cmd line argument
    if args.n == -1:
        frames_range = range(depth_frames.shape[0])
    else:
        frames_range = range(args.n, args.n+1)

    main(args.jobname, depth_frames, bgr_frames, frames_range)
