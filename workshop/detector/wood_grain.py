import cv2
import numpy as np

from workshop.utility import CLAHEfilter

sm_blur = np.ones((3,3),dtype="float") * (1.0 / (3*3))

hp_vert = np.array((
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1]), dtype="int")

d1_vert = np.array((
    [-1, 4, -1, -1, -1],
    [-1, 4, -1, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, -1, 4, -1],
    [-1, -1, -1, 4, -1]), dtype="int")

d2_vert = np.array((
    [-1, -1, -1, 4, -1],
    [-1, -1, -1, 4, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, 4, -1, -1, -1],
    [-1, 4, -1, -1, -1]), dtype="int")

d3_vert = np.array((
    [4, -1, -1, -1, -1],
    [4, -1, -1, -1, -1],
    [-1, 4, -1, -1, -1],
    [-1, 4, -1, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, -1, 4, -1],
    [-1, -1, -1, 4, -1],
    [-1, -1, -1, -1, 4],
    [-1, -1, -1, -1, 4]), dtype="int")

d4_vert = np.array((
    [-1, -1, -1, -1, 4],
    [-1, -1, -1, -1, 4],
    [-1, -1, -1, 4, -1],
    [-1, -1, -1, 4, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, 4, -1, -1, -1],
    [-1, 4, -1, -1, -1],
    [4, -1, -1, -1, -1],
    [4, -1, -1, -1, -1]), dtype="int")


kernels = [hp_vert, d1_vert, d2_vert, d3_vert, d4_vert]

clahe = CLAHEfilter(3,6)

def extract_grain(img, horizontal=True):
    gray = clahe.bw_filter(img)
    h,w = gray.shape[:2]
    gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
    # gray = cv2.equalizeHist(gray)
    # edge.calibrate(gray)
    # # gray = cv2.blur(gray,(3,3))
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    # thresh = cv2.bitwise_not(thresh)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3), iterations=1)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (3,3), iterations=1)
    cv2.imshow("grey", gray)
    cv2.waitKey(0)

    grain = np.zeros(gray.shape, dtype=np.uint8)
    for kernel in kernels:
        if horizontal:
            kernel = kernel.T
        output = cv2.filter2D(gray, -1, kernel)
        grain = cv2.add(grain, output)
    grain = cv2.morphologyEx(grain, cv2.MORPH_OPEN, (3,3), iterations=1)
    grain = cv2.morphologyEx(grain, cv2.MORPH_CLOSE, (3,3), iterations=1)

    grain = cv2.resize(grain, (w,h), interpolation=cv2.INTER_NEAREST)
    return grain
