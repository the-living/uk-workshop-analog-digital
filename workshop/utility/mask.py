from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from workshop.utility import Shape

kernel = {
    "SQUARE": cv2.MORPH_RECT,
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "CROSS": cv2.MORPH_CROSS
}

class Mask(object):
    """Generic mask class, containing binary mask"""

    def __init__(self, mask):
        # assert mask.shape[2] == 1 and mask.dtype == 'uint8'
        self.mask = mask
        self.set_kernel()


    def set_kernel(self, size=5, shape="SQUARE"):
        """Set Mask morphology kernel to given size and shape"""
        self.kernel = cv2.getStructuringElement(kernel[shape],(size, size))
    

    # - - - - - - - - - - - - - - - - - - #
    # MORPHOLOGY TRANSFORMATIONS
    # - - - - - - - - - - - - - - - - - - #

    def morph_open(self, update=False, iterations=1):
        """Open (Erode -> Dilate) to remove external noise"""
        if self.mask is None:
            return
        morph = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel, iterations=iterations)
        if not update:
            return morph
        self.mask = morph
    
    def morph_close(self, update=False, iterations=1):
        """Close (Dilate->Erode) to remove internal noise"""
        if self.mask is None:
            return
        morph = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.kernel, iterations=iterations)
        if not update:
            return morph
        self.mask = morph
    
    def morph_dilate(self, update=False, iterations=1):
        """Grow mask around perimeter"""
        if self.mask is None:
            return
        morph = cv2.dilate(self.mask, self.kernel, iterations=iterations)
        if not update:
            return morph
        self.mask = morph
    
    def morph_erode(self, update=False, iterations=1):
        """Erode mask around perimeter"""
        if self.mask is None:
            return
        morph = cv2.dilate(self.mask, self.kernel, iterations=iterations)
        if not update:
            return morph
        self.mask = morph
    
    def invert(self, update=False):
        inverted = cv2.bitwise_not(self.mask)
        if not update:
            return inverted
        self.mask = inverted
    
    # - - - - - - - - - - - - - - - - - - #
    # APPLY MASK
    # - - - - - - - - - - - - - - - - - - #

    def knockout(self, image):
        return cv2.bitwise_and(image, image, mask=self.mask)

    # - - - - - - - - - - - - - - - - - - #
    # VIZ
    # - - - - - - - - - - - - - - - - - - #

    def display(self, window="MASK"):
        cv2.imshow(window, self.mask)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
