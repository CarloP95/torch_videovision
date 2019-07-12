import numpy as np
import PIL
import torch

import os
from glob import glob

import skvideo.io
from skvideo.measure import viideo_score, strred
from torch_videovision.videotransforms.utils import images as imageutils


class LongestVideo_UCF_101(object):
    """ 
    Gets only the longest video from all of the possible videos that starts with the same name.
    Videos with same name in UCF_101 are shards of the same longest video.
    """

    def __call__(self, paths):
        """
        Args: 
            paths: video paths
        """
        
        if not isinstance(paths, list):
            raise TypeError(f'Expected {type(list)}. But got {type(paths)}')

        checkedVideos   = []
        toReturnVideos  = []
        for idx, path in enumerate(paths):

            if idx % 100 == 0 and idx != 0:
                print(f'\r[{LongestVideo_UCF_101}] Processed {idx} videos out of {len(paths)}. Find {len(toReturnVideos)} that fits.', end = '')

            if path[0: -8] in checkedVideos:
                continue
            
            toReturnVideos.append( LongestVideo_UCF_101.getLongestVideoOfSameClass(path, paths) )
            checkedVideos.append(path[0: -8])
        
        return toReturnVideos

    @staticmethod
    def getLongestVideoOfSameClass(videoPath, videoPaths):

        currentVideoName        = os.path.split(videoPath)[1]
        startOfCurrentVideoName = currentVideoName[0: -8]
        savedStats              = {videoPath: skvideo.io.vread(videoPath).shape[0]}

        for videoname in videoPaths:

            videonameSplit = os.path.split(videoname)[1]
            if videonameSplit.startswith(startOfCurrentVideoName):
                savedStats[videoname] = skvideo.io.vread(videoname).shape[0]

        return max(savedStats)