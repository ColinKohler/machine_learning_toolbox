import numpy as np
import cv2

class ImageAugmenter(object):
    def __init__(self, config=None):
        self.config = config

    # Augment the image using the config specified at init
    def augmentImage(self, img, bbox=None):
        return self.augmentImageWithConfig(img, self.config, bbox)

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

    # Crop random window of given size from image
    def cropImageRandomWindow(self, img, window_size, bbox=None):
        height, width, channels = img.shape
        start_x = self._randIntInRange((0,width-window_size))
        start_y = self._randIntInRange((0,height-window_size))
        img = img[start_y:start_y+window_size, start_x:start_x+window_size,:]

        height, width, channels = img.shape
        if bbox is not None:
            min_bbox = [max(bbox[0][0] - start_x, 0), max(bbox[0][1] - start_y, 0)]
            max_bbox = [min(bbox[1][0] - start_x, width), min(bbox[1][1] - start_y, height)]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Crop the desired pixels from the left and right sides
    def cropImageLR(self, img, pixels, bbox=None):
        img = img[:,pixels:-pixels,:]
        height, width, channels = img.shape

        if bbox is not None:
            min_bbox = [max(bbox[0][0] - pixels, 0), bbox[0][1]]
            max_bbox = [min(bbox[1][0] - pixels, width), bbox[1][1]]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Crop the desired pixels from the top and bottom sides
    def cropImageTB(self, img, pixels, bbox=None):
        img = img[pixels:-pixels,:,:]
        height, width, channels = img.shape

        if bbox is not None:
            min_bbox = [bbox[0][0], max(bbox[0][1] - pixels, 0)]
            max_bbox = [bbox[1][0], min(bbox[1][1] - pixels, height)]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Take a random crop from the imag
    def cropImageRandom(self, img, crop_range, bbox=None):
        top_crop    = self._randIntInRange(crop_range[0])
        bottom_crop = self._randIntInRange(crop_range[1])
        left_crop   = self._randIntInRange(crop_range[2])
        right_crop  = self._randIntInRange(crop_range[3])

        height, width, channels = img.shape
        bottom_crop_tmp = height if bottom_crop == 0 else -bottom_crop
        right_crop_tmp = width if right_crop == 0 else -right_crop

        img = img[top_crop:bottom_crop_tmp,left_crop:right_crop_tmp,:]
        height, width, channels = img.shape
        if bbox is not None:
            min_bbox = [max(bbox[0][0] - left_crop, 0), max(bbox[0][1] - top_crop, 0)]
            max_bbox = [min(bbox[1][0] - left_crop, width), min(bbox[1][1] - top_crop, height)]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Resize the image to the given shape
    def resizeImage(self, img, new_dim, bbox=None):
        old_height, old_width, channels = img.shape
        img = cv2.resize(img, new_dim)
        height, width, channels = img.shape

        if bbox is not None:
            x_scale = width / float(old_width)
            y_scale = height / float(old_height)
            min_bbox = [bbox[0][0] * x_scale, bbox[0][1] * y_scale]
            max_bbox = [bbox[1][0] * x_scale, bbox[1][1] * y_scale]

            return img, [min_bbox, max_bbox]
        else:
            return img

    # Flip a image
    def flipImageLR(self, img, prob, bbox=None):
        if np.random.uniform(0.0, 1.0) >= prob:
            img = cv2.flip(img, 1)

            if bbox is not None:
                height, width, channels = img.shape
                min_to_side = [width - bbox[0][0], bbox[0][1]]
                max_to_side = [width - bbox[1][0], bbox[1][1]]

                return img, [min_to_side, max_to_side]
            else:
                return img
        else:
            if bbox is not None:
                return img, bbox
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
            min_bbox = H.dot(self._createHPoint(bbox[0]))
            max_bbox = H.dot(self._createHPoint(bbox[1]))
            min_bbox = [round(x) for x in min_bbox]
            max_bbox = [round(x) for x in max_bbox]
            bbox = [min_bbox, max_bbox]

            return img, bbox
        else:
            return img

    # Add random noise to the image
    def addNoiseToImage(self, img, min_scale, max_scale, mode='gaussian', bbox=None):
        scale = np.random.uniform(min_scale, max_scale)
        noise = np.zeros(img.shape, np.uint8)
        if mode == 'gaussian':
            cv2.randn(noise, np.zeros(3), np.ones(3)*255*scale)

        img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

        if bbox is not None:
            return img, bbox
        else:
            return img

    # Add jitteer to the RGB channels
    def addColorJitter(self, img, jitter_range, bbox=None):
        r_jitter = self._randIntInRange(jitter_range)
        g_jitter = self._randIntInRange(jitter_range)
        b_jitter = self._randIntInRange(jitter_range)

        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
        img = np.dstack((
            np.roll(R, r_jitter, axis=0),
            np.roll(G, g_jitter, axis=1),
            np.roll(B, b_jitter, axis=0)
            ))

        if bbox is not None:
            return img, bbox
        else:
            return img

    # Alter the intensities of the RGB channels in the image
    def alterRGBIntensities(self, img, bbox=None):
        pass

    # Create 2D homogenous point
    def _createHPoint(self, p):
        return [p[0], p[1], 1.0]

    # Generate random int in the given range
    def _randIntInRange(self, r):
        return np.random.randint(r[0], r[1])
