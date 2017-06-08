import json
import cv2
import random
import numpy as np

class RigorPerceptImporter(object):
    def __init__(self, metadata_path, num_output, batch_size, mean, one_hot_encoding=None):
        self.num_output = num_output
        self.batch_size = batch_size
        self.one_hot_encoding = one_hot_encoding
        self.mean = mean
        self.pointer = 0

        self.readMetadata(metadata_path)
        self.shuffleData()

    # Read percept data from metadata json file
    def readMetadata(self, path):
        with open(path, 'r') as f:
            self.percepts = json.load(f)
        self.num_percepts = len(self.percepts)

    # Get the next batch
    def getBatch(self, domain=None):
        batch_percepts = self.percepts[self.pointer:self.pointer + self.batch_size]
        labels = np.zeros([self.batch_size, self.num_output])
        images = np.ndarray([self.batch_size, 227, 227, 3])
        for i, percept in enumerate(batch_percepts):
            # Load image
            img_path = percept['locator']
            img = cv2.imread(img_path.replace('file://', ''))
            img = cv2.resize(img, (227, 227))
            img = img.astype(np.float32)
            img -= self.mean
            images[i] = img

            # Get annotation
            annotations = percept['annotations']
            annotation = self.getAnnotationWithDomain(annotations, domain)
            if self.one_hot_encoding:
                labels[i][self.one_hot_encoding[annotation['model']]] = 1.0
            else:
                labels[i] = self.getBBoxFromAnnotation(annotation)

        return images, labels

    # Get the annotation with the given domain or the first annotation if no domain is given
    def getAnnotationWithDomain(self, annotations, domain):
        try:
            idx = next(idx for (idx, d) in enumerate(annotations) if d['domain'] == domain) if domain else 0
        except StopIteration:
            print 'Did not find any annotations labels with domain: {}'.format(domain)
        return annotations[idx]

    # Get the bounding box as [x1, y1, x2, y2]
    def getBBoxFromAnnotation(self, annotation):
        bbox = annotations['boundary']
        return [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

    # Shuffles data if desired
    def shuffleData(self):
        random.shuffle(self.percepts)

    # Reset percepts and pointer
    def resetPointer(self):
        self.pointer = 0
        self.shuffleData()

