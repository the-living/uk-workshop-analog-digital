from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

import json

from workshop.utility import Shape
from workshop.utility import Blob

from workshop.utility import ColorRange
from workshop.utility import Mask

from workshop.utility import corner_transform
from workshop.utility import ordered_points

class MarkerCrop(object):
    """Crop to tape markers of known color"""

    def __init__(self, min_color, max_color, min_area=200, max_area=2000, inset=0):
        self.min_area = min_area
        self.max_area = max_area
        self.shape_sides = 4
        self.inset = inset

        self.color = ColorRange(min_color, max_color)

        self.detector = Blob(self.min_area, self.max_area)

        self.corners = []

        self.calibrated = False
    
    @property
    def corners_cnt(self):
        """Returns corners as a cv2 contour"""
        return np.array([[pt] for pt in self.corners], dtype='int')
    
    def load(self, fp):
        with open(fp, mode='r') as f:
            self.corners = json.load(f)
        f.close()

    def detect(self, img, center=False):
        if not self.calibrated:
            self.color.calibrate(img)
        
        mask = Mask(self.color.detect(img))

        mask.morph_open(update=True)
        mask.morph_close(update=True)

        if center:
            self.corners = self.get_center_points(mask.mask)
        else:
            self.corners = self.get_corner_points(mask.mask)
    
    # def get_marker_mask(self, img):
    #     """
    #     Searches an image for markers of a given color
    #     used to define crop region
    #     and returns mask
    #     """
    #     #BLUR THE IMAGED WITH BILATERAL FILTER TO RETAIN EDGES
    #     denoise = cv2.bilateralFilter(img, 50, 75, 75)
    #     #CONVERT IMAGE FROM RGB TO HSV FOR COLOR DETECTION
    #     denoise = cv2.cvtColor(denoise, cv2.COLOR_BGR2HSV)
    #     #CREATE MASK FROM REGIONS OF IMAGE WITHIN COLOR RANGE
    #     mask = cv2.inRange(denoise, self.min_color, self.max_color)
    #     #DILATE AND ERODE MASK TO ELIMINATE HOLES
    #     mask = cv2.dilate(mask, None, iterations=4)
    #     mask = cv2.erode(mask, None, iterations=4)
    #     return mask
    
    def get_corner_points(self, mask, display=False):
        #CREATE CONTOURS FROM MASK
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        #FILTER CONTOUR SHAPES BY SIZE AND NUMBER OF SIDES
        shapes = [Shape(cnt) for cnt in contours if self.min_area < cv2.contourArea(cnt) < self.max_area]
        shapes = [shape for shape in shapes if shape.sides == self.shape_sides]

        if len(shapes) != 4:
            print("CORNERS DETECTED: {} ...INVALID RESULTS".format(len(shapes)))
            for shape in shapes:
                print(shape.shape)
                mask_viz = mask.copy()
                mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(mask_viz, [shape.contour], -1, (0,255,0), 2)
                print( shape.approx)
                for x,y in shape.as_tuples(shape.approx_poly()):
                    cv2.circle(mask_viz, (x,y), 3, (0, 0, 255), -1)
                cv2.imshow("CORNER VIZ", mask_viz)
                cv2.waitKey(0)
            cv2.destroyWindow("CORNER VIZ")
            return None
        
        #RETURN INNER CORNERS OF DETECTED CROP SHAPES
        h,w = mask.shape[:2]
        return ordered_points([shape.closest_corner((w//2,h//2)) for shape in shapes])

    
    def get_center_points(self, mask):
        #DETECT BLOBS IN MASK
        keypoints = self.detector.detect(mask)
        return ordered_points([kp.pt for kp in keypoints])
    
    def mask_image(self, img):
        """
        Masks a given image by crop points
        """
        if len(self.corners) < 4:
            self.detect(img)
        #GET DIMENSIONS OF SOURCE IMAGE
        h,w = img.shape[:2]
        #CREATE BLANK MASK AT SAME DIMENSION
        mask = np.zeros((h,w,1), dtype=np.uint8)
        #DRAW MASK SHAPE FROM POINTS
        cv2.drawContours(mask, [self.corners_cnt], -1, (255), -1)
        return cv2.bitwise_and(img, img, mask=mask)
    
    def crop_image(self, img, output_dim=None):
        if len(self.corners) < 4:
            self.detect(img)
        return corner_transform(img, self.corners, output_dim)
    
    def preview(self, img):
        if len(self.corners) < 4:
            self.detect(img)
        
        preview_img = img.copy()

        cv2.drawContours(preview_img, [self.corners_cnt], -1, (0,255,0), 2)
        for x,y in self.corners:
            cv2.circle(preview_img, (x,y), 5, (0, 0, 255), -1)

        cv2.imshow("CROP PREVIEW", preview_img)
        k = cv2.waitKey(0)
        if chr(k & 255) == 'x':
            print("RE-DETECTING...")
            self.detect(img)
            return False
        cv2.destroyWindow("CROP PREVIEW")
        return True








