from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from workshop.utility import point_distance

def ordered_points(points):
    # print(points, points.shape)

    rect = np.zeros((4,2), dtype=np.float32)
    s = np.sum(points, axis=1)
    # print( s, np.argmax(s))
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def corner_transform(image, points, dim=None):
    rect = ordered_points(points)
    if dim is None:
        (tl, tr, br, bl) = rect
        w = int(max(point_distance(br,bl), point_distance(tr, tl)))
        h = int(max(point_distance(tl,bl), point_distance(tr,br)))
        # w = max(l1,l2)
        # h = min(l1,l2)        
    else:
        w,h = dim

    dst = np.array([
        [0,0],
        [w-1,0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w,h))

def rotate_img(image, angle):
    h,w = image.shape[:2]
    cx,cy = w//2, h//2

    M = cv2.getRotationMatrix2D((cx,cy), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    w_ = int((h * sin) + (w * cos))
    h_ = int((h * cos) + (w * sin))
    M[0,2] += (w_/2) - cx
    M[1,2] += (h_/2) - cy
    return cv2.warpAffine(image, M, (w_,h_))