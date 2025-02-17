import numpy as np
from PIL import Image
import torch

from torch_videovision.videotransforms.utils import images as imageutils


class TransposeChannels(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a list of numpy.ndarrays of shape (C x m x H x W) in the same range"""

    def __init__(self, reverse = False, channel_nb = 3):
        
        self.channel_nb = channel_nb
        self.reverse    = reverse

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray) and not self.reverse:
            
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
            
        elif isinstance(clip[0], np.ndarray) and self.reverse:
            
            ch, _, h, w = clip.shape 
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
            
        else:
            raise TypeError('Expected numpy.ndarray \
            but got list of {0}'.format(type(clip[0])))
        
        np_clip = None
        # Convert
        if self.reverse:
            np_clip = clip.transpose(1, 2, 3, 0)
        else:
            np_clip = clip.transpose(3, 0, 1, 2)

        return np_clip


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = imageutils.convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
                
            return tensor_clip


class ToTensor(object):
    """Converts numpy array to tensor
    """

    def __call__(self, array):
        tensor = torch.from_numpy(array)
        return tensor.float()
