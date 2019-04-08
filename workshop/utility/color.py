from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from workshop.utility import Mask


class ColorRange(object):
    """Color detection class"""
    def __init__(self, color_min, color_max):
        self.color = list(color_min) + list(color_max)
        self.val = 0
    
    @property
    def color_min(self):
        return np.array(self.color[:3])
    
    @property
    def color_max(self):
        return np.array(self.color[3:])

    @property
    def string(self):
        label = "Min( "
        for i,val in enumerate(self.color):
            if i % 3 == 0 and i > 0:
                label += ") Max( "
            if i == self.val:
                start = len(label)
            label += str(val)
            label += " "
            if i == self.val:
                end = len(label)
        label += ")"
        return label, start, end
        
    
    def detect(self, img):
        """
        Searches an image for markers of a given color
        used to define crop region
        and returns mask
        """
        #BLUR THE IMAGED WITH BILATERAL FILTER TO RETAIN EDGES
        denoise = cv2.bilateralFilter(img, 11, 17, 17)
        #CONVERT IMAGE FROM RGB TO HSV FOR COLOR DETECTION
        denoise = cv2.cvtColor(denoise, cv2.COLOR_BGR2HSV)
        #CREATE MASK FROM REGIONS OF IMAGE WITHIN COLOR RANGE
        return cv2.inRange(denoise, self.color_min, self.color_max)
    
    def calibrate(self, img):
        window_label = "COLORRANGE Calibration"
        while True:
            # DRAW SETTING ON IMAGE
            calib = cv2.cvtColor(self.detect(img), cv2.COLOR_GRAY2BGR)
            label, start, end = self.string
            cv2.putText(calib, label[:start], (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            offset = cv2.getTextSize(label[:start], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            sub_label = label[start:end]
            cv2.putText(calib, sub_label, (offset,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            offset = cv2.getTextSize(label[:end], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            label = label[end:]
            cv2.putText(calib, label, (offset, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            # SHOW CURRENT EGDE DETECTION
            cv2.imshow(window_label, calib)

            # WAIT FOR KEYBOARD INPUT
            k = cv2.waitKeyEx(0)
            if k in (27, 13, 32):
                """ESC/SPACE/ENTER - Exit calibration"""
                break
            if k == 2490368:
                """Up Arrow -- Increase level"""
                self.color[self.val] = (self.color[self.val] + 5) & 255
            if k == 2621440:
                """Down Arrow -- Decrease level"""
                self.color[self.val] = (self.color[self.val] - 5) & 255
            if k == 2424832:
                """Left Arrow -- Shift pointer left"""
                self.val = (self.val - 1) % len(self.color)
            if k == 2555904:
                """Right Arrow - Shift pointer right"""
                self.val = (self.val + 1) % len(self.color)
            
        # KILL CALIBRATION WINDOW
        cv2.destroyWindow(window_label)
