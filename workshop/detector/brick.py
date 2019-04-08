from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from math import fabs

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
from workshop.utility import corner_transform
from workshop.utility import rotate_img

Hole = namedtuple('Hole',['x', 'y', 'r'])

class Brick(object):
    """Brick class"""
    def __init__(self, img, x, y, r, w, h):
        self.x = x
        self.y = y
        self.rotation = r
        self.width = w
        self.height = h
        self.face = img
        if self.width > self.height:
            self.face = rotate_img(self.face, 90)
        self.holes = self.detect_holes()
        self.precision = 3
    
    def __repr__(self):
        return "Brick(x:{1:.{0}f} y:{2:.{0}f} w:{3:.{0}f} h:{4:.{0}f} rot:{5:.{0}f})".format(self.precision, self.x, self.y,
            self.width, self.height, self.rotation)
    
    def rotate_crop(self, img):
        return None
    
    def detect_holes(self):
        return None
    
    def dump(self):
        data = {
            'cx': self.x,
            'cy': self.y,
            'width': self.width,
            'height': self.height,
            'rotation': self.rotation
        }
        return data

class BrickDetector(object):
    """Brick detection class"""

    def __init__(self, color_min=(15, 0, 25), color_max = (220,255,255), min_size=20000, pixel_mm=2.0):
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
    
    def dump(self, fp):
        data = [b.dump() for b in self.bricks]
        with open(fp, mode='w') as f:
            json.dump(data, f)
        f.close()

    def detect(self, img, draw=True, show=False, pause=False, verbose=False):
        if verbose:
            print("DETECTING...")
        if not self.calibrated:
            self.calibrate(img)
        mask = cv2.bitwise_not(self.color.detect(img))

        annotated = img.copy()

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, self.min_size)
        cnts = sort_contour(cnts)
        
        self.bricks = []

        for c in cnts:
            if cv2.contourArea(c) < self.min_size:
                continue
            
            # original = img.copy()
            box = cv2.minAreaRect(c)
            (bx,by), (bw,bh), br = box
            box = cv2.boxPoints(box)
            box = np.array(box, dtype='int')
            box = ordered_points(box)

            if draw:
                cv2.drawContours(annotated, [box.astype("int")], -1, (0,255,0), 2)
                for x,y in box:
                    cv2.circle(annotated, (x,y), 5, (0, 0, 255), -1)
            
            if fabs(br) > 45:
                bw,bh = bh,bw
                br += 90
            self.bricks.append(Brick(corner_transform(img,box,(int(bw),int(bh))),bx,by,br,bw,bh))

            if show:
                cv2.imshow('img', original)
            

                # otsu = cv2.GaussianBlur(cv2.cvtColor(bricks[-1].face, cv2.COLOR_BGR2GRAY), (5,5), 0)
                # otsu = cv2.threshold(otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # otsu = cv2.threshold(cv2.cvtColor(bricks[-1].face, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
                # self.thresh.calibrate(bricks[-1].face)
                # t = self.thresh.detect(bricks[-1].face)
                # brick_cnt = cv2.findContours(cv2.bitwise_not(t), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][-1]

                
                # cv2.imshow("BRICK", label_contours(bricks[-1].face, [brick_cnt]))
                brick_face = self.bricks[-1].face
                # if bw > bh:
                #     brick_face = rotate_img(brick_face, 90)

                cv2.imshow("BRICK", brick_face)

                c = cv2.waitKey(100)
                cv2.destroyWindow("BRICK")
        
        if verbose:
            for brick in self.bricks:
                print(brick)
        self.dump("detected_bricks.json")
        
        if show and pause:
            cv2.waitKey(0)
        
        if show:
            cv2.destroyWindow('img')
        
        return annotated

        

    

    
