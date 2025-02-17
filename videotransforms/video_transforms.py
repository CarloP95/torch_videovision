import numbers
import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy
import torch
import torchvision

from . import functional as F

from torchvision.transforms import ToPILImage, ToTensor


class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return np.array([np.fliplr(img) for img in clip])
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = np.array(F.resize_clip(
            clip, self.size, interpolation=self.interpolation))
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds

    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.misc.imrotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list or ndarray): list of PIL.Image or ndarray

        Returns:
        list PIL.Image or ndarray : list of transformed PIL.Image or ndarray if input is of this type.
        """
        isNdArray = isinstance(clip[0], np.ndarray)
        
        converter = ToPILImage()
        unconverter = ToTensor()
        
        if isNdArray:
            clip = [converter(frame) for frame in clip]
        
        if isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            raise TypeError(f'Trying to convert from {type(clip[0])} format that is not supported.')

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []; jittered_img = None
        
        for img in clip:
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(jittered_img)
        
        if isNdArray:
            
            convertedClip = [] 
            for jitteredFrame in jittered_clip:
                
                convertedClip.append( np.array(jitteredFrame, dtype=np.uint8).reshape(jitteredFrame.size[1], jitteredFrame.size[0], 3) )
            
            jittered_clip = np.array(convertedClip)
            
        return jittered_clip


class TemporalCrop(object):
    """Extract temporal crop for frames in video.

    Args:
    
    numFrames: int
        numFrames to crop.
    
    start_frame_idx: int, default 0
        start_frame index to begin cropping. 
    """

    def __init__(self, numFrames, start_frame_idx = 0):
        
        assert isinstance(numFrames, int)
        assert isinstance(start_frame_idx, int)
        
        self.numFrames = numFrames
        self.start_frame_idx = start_frame_idx

    def __call__(self, clip):
        """
        Args:
        img (numpy.ndarray): List of images to be cropped
        in format (#frames, h, w, channels) in numpy.ndarray

        Returns:
        numpy.ndarray: Cropped list of images
        """
        
        return clip[ : , self.start_frame_idx : self.start_frame_idx + self.numFrames, :, : ]


class RandomTemporalCrop(object):
    """Extract random temporal crop for frames in video.

    Args:
    
    numFrames: int
        numFrames to crop.
    
    """

    def __init__(self, numFrames):
        
        assert isinstance(numFrames, int)
        
        self.numFrames = numFrames


    def __call__(self, clip):
        """
        Args:
        img (numpy.ndarray): List of images to be cropped
        in format (#frames, h, w, channels) in numpy.ndarray

        Returns:
        numpy.ndarray: Cropped list of images
        """
        #try:
            #print(f"RandomTemporalCrop: {clip.shape}")
            
        #except AttributeError as _:
            #print(f"The clip is not a ndarray neither a tensor: {type(clip)}, {len(clip)}")
            
        random_start_frame_idx = -1
        
        try:
            random_start_frame_idx = np.random.randint(0, clip.shape[1] - self.numFrames)
        
        except ValueError as _:
            random_start_frame_idx = 0 #Video has the exact number of frames requested.
        
        return clip[ : , random_start_frame_idx : random_start_frame_idx + self.numFrames, :, : ]
