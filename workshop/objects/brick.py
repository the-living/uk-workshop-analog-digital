from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json
from collections import namedtuple

from workshop.objects import Detection

Hole = namedtuple('Hole',['x', 'y', 'r', 'cnt'])

def create_hole(x, y, r, cnt):
    contour = [(cx,cy) for cx,cy in np.vstack(cnt).squeeze().tolist()]
    return Hole(x, y, r, contour)

class Brick(Detection):
    """Detected Brick Class"""
    def __init__(self, x, y, l, w, r, bound, holes):
        super(Brick, self).__init__(x, y, l, w, r, bound)
        self.holes = [create_hole(*h) for h in holes]
    
    def __repr__(self):
        return "Brick(x:{1:.{0}f} y:{2:.{0}f} l:{3:.{0}f} w:{4:.{0}f} rot:{5:.{0}f} holes:{:d})".format(
            self.precision, self.x, self.y,
            self.length, self.width, self.rotation,
            len(self.holes))
    
    def dump(self):
        data = {
            'cx': self.x,
            'cy': self.y,
            'length': self.length,
            'width': self.width,
            'depth': self.depth,
            'rotation': self.rotation,
            'boundary': self.boundary,
            'holes': self.holes
        }
        return data
