import warnings
from skimage import transform
import numpy as np

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['kp']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h = max(h, new_h)
        new_w = max(w, new_w)

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h, left: left + new_w]
        keypoints[:, 0] -= left
        keypoints[:, 1] -= top

        return {'image': image, 'kp': keypoints}
