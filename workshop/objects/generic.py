from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json

class Detection(object):
    """Generic detected object class"""
    def __init__(self, x, y, l, w, r, bound):
        self.x = x
        self.y = y
        self.length = l
        self.width = w
        self.depth = -1
        self.rotation = r
        self.boundary = bound
        self.precision = 3

    @property
    def polyline(self):
        return 
    
