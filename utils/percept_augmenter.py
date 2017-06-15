import numpy as np
import cv2

class ImageAugmenter(object):
    def __init__(self):
        pass

    # Augment the image using the given config dict
    def augmentImageWithConfig(self, img, config, bbox=None):
        for augment in config:
            if bbox is not None:
                img, bbox = augment['method'](img, *augment['args'], bbox=bbox)
            else:
                img = augment['method'](img, *augment['args'])
        if bbox is not None:
            return img, bbox
        else:
            return img

    # Crop the desired pixels from the left and right sides
    def cropImageLR(self, img, pixels, bbox=None):
        img = img[:,pixels:-pixels,:]
        height, width, channels = img.shape

        if bbox is not None:
            min_bbox = [max(bbox[0][0] - pixels, 0), bbox[0][1]]
            max_bbox = [max(bbox[1][0] - pixels, width), bbox[1][1]]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Crop the desired pixels from the top and bottom sides
    def cropImageTB(self, img, pixels, bbox=None):
        img = img[pixels:-pixels,:,:]
        height, width, channels = img.shape

        if bbox is not None:
            min_bbox = [bbox[0][0], max(bbox[0][1] - pixels, 0)]
            max_bbox = [bbox[1][0], max(bbox[1][1] - pixels, height)]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Resize the image to the given shape
    def resizeImage(self, img, new_dim, bbox=None):
        rgb_frame = cv2.resize(rgb_frame, (256, 256))
        if bbox is not None:
            min_bbox = [bbox.[0][0] / 2, bbox.min[0][1] / 2]
            max_bbox = [bbox.[1][0] / 2, bbox.max[1][1] / 2]

            return rgb_frame, [min_bbox, max_bbox]
        else:
            return rgb_frame

    # Flip a image
    def flipImageLR(self, img, bbox=None):
        img = cv2.flip(img, 1)
        if bbox is not None:
            height, width, channels = img.shape
            min_to_side = [width - bbox[0][0] / 2, bbox[0][1]]
            max_to_side = [width - bbox[1][0] / 2, bbox[1][1]]

            return img, [min_to_side, max_to_side]
        else:
            return img


    # Transform the image a random amount
    def transformImage(self, img, min_trans, max_trans, min_rot, max_rot, bbox=None):
        rot = np.random.uniform(min_rot, max_rot)
        trans_x = np.random.uniform(min_trans, max_trans)
        trans_y = np.random.uniform(min_trans, max_trans)

        # Create transfromation matrix
        height, width, channels = img.shape
        center = (height / 2, width / 2)
        H = cv2.getRotationMatrix2D(center, rot, 1.0)
        H[0,-1] += trans_x
        H[1,-1] += trans_y

        # Apply transformation
        img = cv2.warpAffine(img, H, (height, width))
        if bbox is not None:
            min_bbox = H.dot(self.createHPoint(bbox[0]))
            max_bbox = H.dot(self.createHPoint(bbox[1]))
            min_bbox = [round(x) for x in min_bbox]
            max_bbox = [round(x) for x in max_bbox]
            bbox = [min_bbox, max_bbox]

            return img, bbox
        else:
            return img

    # Add random noise to the image
    def addNoiseToImage(self, img, mode='gaussian', bbox=None):
        noise = np.zeros(img.shape, np.uint8)
        if mode == 'gaussian':
            cv2.randn(noise, np.zeros(3), np.ones(3)*255.0.1)

        img = cv2.add(img, noise, dtype=cv2.CV_8UC3)
        return img

    # Take a random crop from the imag
    def cropImageRandom(self, img, zoom, bbox=None):
        pass

    # Create 2D homogenous point
    def _createHPoint(self, p):
        return [p[0], p[1], 1.0]
