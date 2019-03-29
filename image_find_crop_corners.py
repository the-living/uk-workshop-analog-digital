import cv2
import numpy as np
import os
import time

from utility.calibration import CircleGrid

from utility.shape import Shape

from utility.transform import corner_transform
from utility.transform import ordered_points

# img_path = r"capture\cap_1553718048.02017.jpg"
# img_path = r"capture\cap_1553719238.4823034.jpg"
# img_path = r"capture\cap_1553717996.9982228.jpg"
img_path = r"capture\cap_1553719975.5986345.jpg"


crop_w = 1220
crop_h = 915

cb = CircleGrid()
cb.load_calibration("circle_calibration.json")

img = cv2.imread(img_path)
img = cb.undistort(img, crop=True)
h,w = img.shape[:2]
img = img[0:h ,100:w-200]
h,w = img.shape[:2]

# COLOR MASK DOMAIN
color_min = np.array([100, 50, 0])
color_max = np.array([120, 200, 255])

#BLUR IMAGE WITH EDGE RETENTION
denoise = cv2.bilateralFilter(img, 9, 21, 21)

#CONVERT COLOR TO HSV
img_hsv = cv2.cvtColor(denoise, cv2.COLOR_BGR2HSV)

#CREATE COLOR MASK
blue_mask = cv2.inRange(img_hsv, color_min, color_max)
# blue_mask = np.uint8(blue_mask.astype('bool'))

#MORPHOLOGY TO ELIMINATE HOLES
blue_mask = cv2.dilate(blue_mask, None, iterations=4)
blue_mask = cv2.erode(blue_mask, None, iterations=4)

#CONTOUR MASK
contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#CONVERT TO SHAPE OBJECTS & FILTER BY SIZE
shapes = [Shape(cnt) for cnt in contours if cv2.contourArea(cnt) > 200]
#FILTER SHAPES BY SIDES
shapes = [shape for shape in shapes if shape.sides == 4]
#GET INNER CORNERS (CLOSEST TO CENTER)
inner_corners = [shp.closest_corner((w//2,h//2)) for shp in shapes]
# inner_corners_ord = ordered_points(inner_corners)

img_anno = img.copy()
for shape, corner in zip(shapes, inner_corners):
    cv2.drawContours(img_anno, shape.approx, -1, (0,0,255), 2)
    cv2.circle(img_anno, corner, 5, (0,255,0), -1)

# for corner in inner_corners_ord:
#     print(corner)
#     cv2.circle(img_anno, tuple(corner), 5, (255,0,0), 2)

cropped = corner_transform(img, inner_corners, (crop_w, crop_h))

# blue_mask = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
# out_img = np.vstack([img, blue_mask])
# out_img = cv2.resize(out_img, None, fx=0.5, fy=0.5)
cv2.imshow('img', img_anno)
cv2.imshow('crop', cropped)
# cv2.imshow('mask', blue_mask)

c = cv2.waitKey(0)
if chr(c % 256) == 'x':
    outdir = "capture"
    t = time.time()
    outfile = "cap-{}.jpg".format(t)
    outfile_crop = "cap-{}.crop.jpg".format(t)
    cv2.imwrite(os.path.join(outdir, outfile), img_anno)
    cv2.imwrite(os.path.join(outdir, outfile_crop), cropped)
cv2.destroyAllWindows()
