import warnings
from skimage import transform
import numpy as np

class ToTensor(object):

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['kp']
        keypoints = np.array(keypoints, dtype=np.float32) / image.shape[0]
        keypoints = keypoints.flatten()
        image = image.transpose((2, 0, 1))
        return {'image': np.array(image, dtype=np.float32),
                'kp': keypoints}