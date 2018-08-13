import numpy as np

from utils.transforms.input_transforms.base_input_transform import BaseInputTransform


class AtariLazyNumpyImageTransform(BaseInputTransform):

    def transform(self, inputs):
        """

        :param inputs: B x H x W x C
        :return: B x C x H x W
        """
        np_arr = np.array(inputs) / 255.0
        assert np_arr.ndim == 3 or np_arr.ndim == 4
        if np_arr.ndim == 3:
            return np_arr.transpose((2, 0, 1))
        return np_arr.transpose((0, 3, 1, 2))
