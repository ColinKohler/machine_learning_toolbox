import json
import cv2
import random
import numpy as np

class PerceptImporter(object):
    def __init__(self, metadata_path, num_output, batch_size, mean, shuffle=True):
        self.shuffle = shuffle
        self.num_output = num_output
        self.batch_size = batch_size
        self.mean = mean
        self.pointer = 0

        self.readMetadata(metadata_path)
        self.shuffleData()

    # Read percept data from metadata json file
    def readMetadata(self, path):
        with open(path, 'r') as f:
            self.percepts = json.load(f)
        self.num_percepts = len(self.percepts)

    # Shuffles data if desired
    def shuffleData(self):
        if self.shuffle:
            random.shuffle(self.percepts)

    # Reset percepts and pointer
    def resetPointer(self):
        self.pointer = 0
        self.shuffleData()

    # Get the next batch
    def getBatch(self, cls=True):
        batch_percepts = self.percepts[self.pointer:self.pointer + self.batch_size]
        labels = np.zeros([self.batch_size, self.num_output])
        images = np.ndarray([self.batch_size, 227, 227, 3])
        for i, percept in enumerate(batch_percepts):
            img_path = percept['locator']
            if len(percept['annotations']) == 0:
                if cls:
                    label = 0
                else:
                    continue
            else:
                if cls:
                    label = 1
                else:
                    bbox = percept['annotations'][0]['boundary']
                    label = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

            # Load image
            img = cv2.imread(img_path.replace('file://', ''))
            img = cv2.resize(img, (227, 227))
            img = img.astype(np.float32)
            img -= self.mean

            images[i] = img
            if cls:
                labels[i][label] = 1
            else:
                labels[i] = label

        return images, labels
