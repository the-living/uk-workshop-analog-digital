from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
from math import fabs

from workshop.utility import point_distance

class Shape(object):
    def __init__(self, contour):
        self.contour = contour
    
    @property
    def area(self):
        return cv2.contourArea(self.contour)
    
    @property
    def approx(self):
        perimeter = cv2.arcLength(self.contour, True)
        return cv2.approxPolyDP(self.contour, 0.04 * perimeter, True)
    
    @property
    def approx_area(self):
        return cv2.contourArea(self.approx)
    
    @property
    def aspect_ratio(self):
        w,h = cv2.boundingRect(self.contour)[2:]
        return max(w,h) / float(min(w,h))
    
    @property
    def sides(self):
        return len(self.approx)

    @property
    def shape(self):
        sides = self.sides
        if sides == 3:
            return "triangle"
        if sides == 4:
            return "square" if fabs(self.aspect_ratio-1) == 0.05 else "rectangle"
        
        ngons = {5:"penta", 6:"hexa", 7:"hepta", 8:"octa", 9:"nona", 10:"deca"}
        if sides in ngons:
            return "{}gon".format(ngons[sides])
        return "{:d}-gon".format(sides)
    
    def approx_poly(self, factor=0.04):
        perimeter = cv2.arcLength(self.contour, True)
        return cv2.approxPolyDP(self.contour, factor * perimeter, True)

    def as_tuples(self, pts):
        stack = np.vstack(self.contour).squeeze().tolist()
        return[(x,y) for x,y in stack]
    
    def approx_tuples(self):
        return self.as_tuples(self.approx)
    
    def closest_corner(self, pt):
        sort_point = sorted(
            self.as_tuples(self.approx),
            key=lambda x: point_distance(pt, x))
        return list(sort_point[0])

class Blob(object):

    def __init__(self, min_blob=25, max_blob=2500):
        self.detector = self.__setup_blob_detector(min_blob, max_blob)

    def __setup_blob_detector(self, min_blob, max_blob):
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minThreshold = 8
        blob_params.maxThreshold = 255
        blob_params.filterByArea = True
        blob_params.minArea = min_blob
        blob_params.maxArea = max_blob
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.1
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.87
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.01
        return cv2.SimpleBlobDetector_create(blob_params)
    
    def detect(self, img):
        return self.detector.detect(img)