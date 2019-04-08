from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

import json

from collections import namedtuple

from workshop.utility import Shape
from workshop.utility import Blob
from workshop.utility import ColorRange
from workshop.utility import Threshold
from workshop.utility import EdgeDetector

from workshop.utility import rotate_img
from workshop.utility import CLAHEfilter
from workshop.utility import white_balance
from workshop.utility import screen

from workshop.utility import filter_contours
from workshop.utility import sort_contour
from workshop.utility import label_contours

from workshop.utility import ordered_points

Hole = namedtuple('Hole',['x', 'y', 'r'])

class Brick(object):
    """Brick class"""
    def __init__(self, l=-1.0, w=-1.0, h=-1.0, holes=[]):
        self.length = l
        self.width = w
        self.height = h
        self.holes = [Hole(*hole) for hole in holes]
    
    def dump(self, filepath):
        data = {
            'length': self.length,
            'width': self.width,
            'height': self.height,
            'holes': [tuple(hole) for hole in self.holes]
        }
        with open(filepath, mode='w') as f:
            json.dump(data, f)
        f.close()

class BrickDetector(object):
    """Brick detection class"""

    def __init__(self, color_min=(15, 0, 25), color_max = (220,255,255), min_size=20000):
        self.color_min = color_min
        self.color_max = color_max
        self.min_size = min_size

        self.bricks_locations = []
        self.bricks = []

        self.color = ColorRange(self.color_min, self.color_max)
        self.thresh = Threshold()
        self.edge = EdgeDetector()

        self.calibrated = False
    
    def __prep_image(self, img, screen=2, wb=True, clahe=True, b_kernel=50, b_sigma=75):
        image = img.copy()
        if b_kernel:
            image = cv2.bilateralFilter(image, b_kernel, b_sigma, b_sigma)
        if wb:
            image = white_balance(image)
        if clahe:
            image = clahe.yuv_filter(image)
        for _ in range(int(screen)):
            image = screen(image)
        return cv2.cvtColor(image, cv2.COLOR_BRG2HSV)

    def calibrate(self, img):
        self.color.calibrate(img)
        self.thresh.calibrate(img)
        self.edge.calibrate(img)
        self.calibrated = True
        pass
    
    def save_calibration(self):
        pass
    
    def load_calibration(self, fp):
        pass

    def detect(self, img):
        if not self.calibrated:
            self.calibrate(img)
        mask = cv2.bitwise_not(self.color.detect(img))

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, self.min_size)
        cnts = sort_contour(cnts)

        # anno = label_contours(img, cnts)
        # cv2.imshow('detected', anno)
        # cv2.waitKey(0)
        # cv2.destroyWindow('detected')

        for c in cnts:
            if cv2.contourArea(c) < self.min_size:
                continue
            
            original = img.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            # box = cv2.cornerSubPix(gray, box, (11,11), (-1,-1), CRITERIA)
            box = np.array(box, dtype='int')
            

            box = ordered_points(box)
            cv2.drawContours(original, [box.astype("int")], -1, (0,255,0), 2)
            for x,y in box:
                cv2.circle(original, (x,y), 5, (0, 0, 255), -1)
            
            print("BOX AREA: {}".format(cv2.contourArea(box.astype("int"))))
            cv2.imshow('img', original)
            c = cv2.waitKey(0)
            if c == 27:
                break
        cv2.destroyWindow('img')
    
    def preview(self, img):
        self.color.calibrate(img)

        

    

    
