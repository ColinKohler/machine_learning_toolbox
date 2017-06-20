import scipy.misc
from utils.image_augmenter import ImageAugmenter

img = scipy.misc.imread('test.jpg')
aug = ImageAugmenter()
crop_range = [(0,200), (0,30), (0,100), (0,100)]
routine = [{'method' : aug.cropImageRandom, 'args' : [crop_range]},
           {'method' : aug.flipImageLR,     'args' : [0.5]},
           {'method' : aug.transformImage,  'args' : [0, 10, -15, 15]},
           {'method' : aug.addNoiseToImage, 'args' : [0.1]}]

for i in range(25):
    aug_img = aug.augmentImageWithConfig(img, routine)
    scipy.misc.imsave('tmp/{}.jpg'.format(i), aug_img)
