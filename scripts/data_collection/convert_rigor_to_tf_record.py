#!/usr/bin/python

import os
import argparse
import json
import hashlib
import copy
import cv2

import tensorflow as tf
from object_detection.utils import dataset_util

from utils.image_augmenter import ImageAugmenter

# Convert a Rigor metadata file to TF Record format
def main():
    parser = argparse.ArgumentParser(description='Convert percepts in Rigor format to Tensorflow Record percepts')
    parser.add_argument('class_encoding_path', type=str,
                        help='Path to metadata file for the onehot encoding')
    parser.add_argument('rigor_metadata_path', type=str,
                        help='Path to Rigor metadata file to convert')
    parser.add_argument('output_path', type=str,
                        help='Path to output TFRecord')
    parser.add_argument('--num_augments', type=int, default=0,
                        help='Number of times to augment the data')
    args = parser.parse_args()

    # Init data
    writer = tf.python_io.TFRecordWriter(args.output_path)
    percepts = readRigorMetadata(args.rigor_metadata_path)
    one_hot_encoding = loadClassEncoding(args.class_encoding_path)

    # Setup image augmenter if neccessary
    if args.num_augments > 0:
        augmenter = ImageAugmenter()
        routine = [{'method' : augmenter.cropImageRandomWindow, 'args' : [400]},
                   {'method' : augmenter.resizeImage,           'args' : [(480,480)]},
                   {'method' : augmenter.flipImageLR,           'args' : [0.5]},
                   {'method' : augmenter.addColorJitter,        'args' : [(0,2)]},
                   {'method' : augmenter.transformImage,        'args' : [-5, 5, -15, 15]},
                   {'method' : augmenter.addNoiseToImage,       'args' : [0.1, 0.1]}
                  ]

    # Convert percept into record, augmenting if desired
    for percept in percepts:
        # TODO: Tmp hold over to remove percepts without bboxes
        if percept['annotations'][0]['boundary'] is None:
            continue

        # Read image and create bounding box as [min(x,y), max(x,y)]
        filename = os.path.splitext(os.path.basename(percept['locator']))[0]
        img = cv2.imread(percept['locator'].replace('file://', ''))
        bbox = percept['annotations'][0]['boundary']
        bbox = [bbox[0], bbox[2]]

        for i in range(args.num_augments+1):
            # Set temp variables correctly
            if i > 0:
                tmp_img, tmp_bbox = augmenter.augmentImageWithConfig(img, routine, bbox)
            else:
                tmp_img = img
                tmp_bbox = bbox

            # Create TF percept and write to record
            tmp_percept = copy.deepcopy(percept)
            tmp_percept['locator'] = filename + '_{}'.format(i)
            tmp_percept['annotations'][0]['boundary'] = tmp_bbox

            if tmp_bbox[0][0] < 0 or tmp_bbox[0][0] > 480 or \
               tmp_bbox[1][0] < 0 or tmp_bbox[1][0] > 480:
                   print tmp_bbox
            tf_percept = createTFPercept(tmp_img, tmp_percept, one_hot_encoding)
            writer.write(tf_percept.SerializeToString())

    writer.close()

# Create TF Record percept
def createTFPercept(img, percept, encoding):
    filename = percept['locator']
    img = cv2.imencode('.jpeg', img)[1].tostring()
    key = hashlib.sha256(img).hexdigest()
    img_format = 'jpeg'

    height = percept['y_size']
    width = percept['x_size']

    xmin = list()
    ymin = list()
    xmax = list()
    ymax = list()
    classes = list()
    classes_text = list()
    for annotation in percept['annotations']:
        bbox = annotation['boundary']

        xmin.append(float(bbox[0][0]) / width)
        ymin.append(float(bbox[0][1]) / height)
        xmax.append(float(bbox[1][0]) / width)
        ymax.append(float(bbox[1][1]) / height)
        classes_text.append(annotation['model'].encode('utf8'))
        classes.append(encoding[annotation['model']])

    tf_percept = tf.train.Example(features=tf.train.Features(feature={
        'image/height' : dataset_util.int64_feature(height),
        'image/width' : dataset_util.int64_feature(width),
        'image/filename' : dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id' : dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256' : dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded' : dataset_util.bytes_feature(img),
        'image/format' : dataset_util.bytes_feature(img_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_percept

# Load the class encoding
def loadClassEncoding(path):
    one_hot_encoding = dict()
    with open(path, 'r') as f:
        for line in f:
            cls, num = line.split(' ', 1)
            one_hot_encoding[cls] = int(num)
    return one_hot_encoding

# Read percept data from metadata json file
def readRigorMetadata(path):
    with open(path, 'r') as f:
        percepts = json.load(f)
    return percepts

if __name__ == '__main__':
    main()
