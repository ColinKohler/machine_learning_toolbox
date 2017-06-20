#!/usr/bin/python

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.image_augmenter import ImageAugmenter
from utils.object_segmenter import ObjectSegmenter

def main(jobname, depth_frames, rgb_frames, frames_range, datapath, n_augs, seg):
    segmenter = ObjectSegmenter(datapath + 'hdf5/camera_info.txt')
    augmenter = ImageAugmenter()

    bbox_areas = list(); percepts = list()
    for frame_num in frames_range:
         # Get frames and convert BGR frame to rgb
        depth_frame = depth_frames[frame_num]
        rgb_frame = cv2.cvtColor(rgb_frames[frame_num], cv2.COLOR_BGR2RGB)

        # Segment the closest object out and create its bounding box
        if seg:
            print "Segmenting Frame {}...".format(frame_num)

            # Get point cloud out of depth frame
            num_points = depth_frame.shape[0] / 3
            cloud = depth_frame.reshape((num_points, 3))

            # Segment bag and convert 3d points to 2d
            bag_bbox = segmenter.segmentObjectInFrame(cloud, 2.35)
            valid, bbox_areas = validate_bbox(bbox, bbox_areas)
            if not valid:
                print "Rejecting frame due to bad bounding box..."
                continue
        else:
            bbox = None

        # Augment data
        print "Augmenting Frame {}...".format(frame_num)
        crop_range = [(0,200), (0,30), (0,100), (0,100)]
        routine = [{'method' : aug.cropImageRandom, 'args' : [crop_range]},
                   {'method' : aug.flipImageLR,     'args' : [0.5]},
                   {'method' : aug.transformImage,  'args' : [0, 10, -15, 15]},
                   {'method' : aug.addNoiseToImage, 'args' : [0.1]}]

        for i in range(n_augs):
            img = aug.augmentImageWithConfig(rgb_frame, routine)
            if (i % 5) < 3:
                dset = 'train'
            elif (i % 5) == 3:
                dset = 'test'
            else:
                dset = 'validate'

            filepath = DATA_PATH + dset + '/{}_{}_{}.jpg'.format(jobname, frame_num, i)
            percept = create_percept(filepath, dset, frame, bbox_tmp)
            percepts.append(percept)
            plot_frame_with_box(frame, bbox_tmp, filepath)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(filepath, img)

    # Save percepts
    percepts_path = datapath + 'metadata/' + jobname + '_percepts.json'
    with open(percepts_path, 'w') as p_file:
        json.dump(percepts, p_file)

# Create the percept annotations for the given image
def create_percept(filepath, dset, frame, bbox):
    percept = dict()
    percept['source'] = 'file://' + filepath
    percept['locator'] = 'file://' + filepath
    percept['format'] = 'image/jpg'
    percept['x_size'] = frame.shape[0]
    percept['y_size'] = frame.shape[1]

    if bbox:
        percept['tags'] = ["Warthog", "trash.bag", dset]

        bbox = [[bbox.min[0], bbox.min[1]],
                [bbox.max[0], bbox.min[1]],
                [bbox.max[0], bbox.max[1]],
                [bbox.min[0], bbox.max[1]]]
        percept['annotations'] = [{'domain' : 'trash',
                                   'model' : 'trash.bag',
                                   'confidence': 1,
                                   'boundary' : bbox,
                                   'annotation_tags' : ['auto.segmented']}]
    else:
        percept['tags'] = ["Warthog", "no.trash.bag", dset]
        percept['annotations'] = [{'domain' : 'trash',
                                   'model' : 'none',
                                   'confidence': 1,
                                   'annotation_tags' : ['auto.segmented']}]
    return percept

# Pad bounding box
def pad_bbox(bbox, pad=25):
    min_bbox = [bbox[0][0], bbox[0][1] - pad]
    max_bbox = [bbox[1][0] + pad, bbox[1][1] + pad]
    return [min_bbox, max_bbox]

# Check if bounding box is a reasonable size
def validate_bbox(bbox, bbox_areas):
    bbox_area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
    mean_bbox_area = np.mean(np.array(bbox_areas)) if bbox_areas else 0

    valid = True
    if len(bbox_areas) < 10:
        bbox_areas.append(bbox_area)
    elif bbox_area < 2 *  mean_bbox_area and bbox_area > 0.25 * mean_bbox_area:
        bbox_areas.append(bbox_area)
    else:
        valid = False

    return valid, bbox_areas

# Save frame with bounding box displayed
def plot_frame_with_box(rgb_frame, bbox, filepath):
    fig, ax = plt.subplots(1)
    ax.imshow(rgb_frame)
    diff_x = bbox[1][0] - bbox[0][0]
    diff_y = bbox[1][1] - bbox[0][1]
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
    parser.add_argument("--n_augs", dest='num_augments', type=int, default=10, help="The desired number of times to augment each frame")
    parser.add_argument("--n_frame", dest='n', type=int, default=-1, help="The desired rgb frame number, or -1 (default) for all")
    parser.add_argument("--seg", dest="seg", action='store_true', help="Set this flag if there are objects to segment")
    parser.add_argument("--no_seg", dest="seg", action='store_false', help="Set this flag if there are no objects to segment")
    parser.set_defaults(seg=True)
    args = parser.parse_args()

    # Get desired frame from number
    h5_file = h5py.File(DATA_PATH + 'hdf5/' + args.h5 + '.h5', 'r')
    depth_frames = h5_file[args.jobname + '_depth']
    bgr_frames = h5_file[args.jobname + '_rgb']

    # Determine range from cmd line argument
    if args.n == -1:
        frames_range = range(depth_frames.shape[0])
    else:
        frames_range = range(args.n, args.n+1)

    main(jobname, depth_frames, rgb_frames, frames_range, datapath, n_augs, seg)
