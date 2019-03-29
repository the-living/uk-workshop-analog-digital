import cv2
import numpy as np

def ordered_points(points):
    rect = np.zeros((4,2), dtype=np.float32)
    s = np.sum(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def corner_transform(image, points, dim):
    rect = ordered_points(points)
    w,h = dim

    dst = np.array([
        [0,0],
        [w-1,0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w,h))