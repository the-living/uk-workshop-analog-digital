import cv2
import numpy as np
from math import fabs

from .geo_util import point_distance

class Shape(object):
    def __init__(self, contour):
        self.contour = contour
    
    @property
    def approx(self):
        perimeter = cv2.arcLength(self.contour, True)
        return cv2.approxPolyDP(self.contour, 0.04 * perimeter, True)
    
    @property
    def sides(self):
        return len(self.approx)

    @property
    def shape(self):
        poly = self.approx
        sides = self.sides
        if sides == 3:
            return "triangle"
        if sides == 4:
            w,h = cv2.boundingRect(poly)[2:]
            aspect = w / float(h)
            return "square" if fabs(aspect-1) == 0.05 else "rectangle"
        
        ngons = {5:"penta", 6:"hexa", 7:"hepta", 8:"octa", 9:"nona", 10:"deca"}
        if sides in ngons:
            return "{}gon".format(ngons[sides])
        return "{:d}-gon".format(sides)
    
    def as_tuples(self, pts):
        stack = np.vstack(self.contour).squeeze().tolist()
        return[(x,y) for x,y in stack]
    
    def closest_corner(self, pt):
        sort_point = sorted(
            self.as_tuples(self.approx),
            key=lambda x: point_distance(pt, x))
        return sort_point[0]

