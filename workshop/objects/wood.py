from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json
from collections import namedtuple

from workshop.objects import Detection


class Wood(Detection):
    """Detected Wood Class"""
    def __init__(self, x, y, l, w, r, bound, grain=None):
        super(Wood, self).__init__(x, y, l, w, r, bound)
        self.grain = grain
    
    def __repr__(self):
        return "Wood(x:{1:.{0}f} y:{2:.{0}f} l:{3:.{0}f} w:{4:.{0}f} rot:{5:.{0}f})".format(
            self.precision, self.x, self.y,
            self.length, self.width, self.rotation)
    
    def dump(self):
        data = {
            'cx': self.x,
            'cy': self.y,
            'length': self.length,
            'width': self.width,
            'rotation': self.rotation,
            'boundary': [(x,y) for x,y in np.vstack(self.boundary).squeeze().tolist()],
            'grain': self.grain
        }
        return data
