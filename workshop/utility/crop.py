from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

import json

from workshop.utility import Shape

from workshop.utility import ColorRange
from workshop.utility import Threshold
from workshop.utility import Mask
from workshop.utility import EdgeDetector

from workshop.utility import corner_transform
from workshop.utility import ordered_points
from workshop.utility import label_contour
from workshop.utility import label_contours
from workshop.utility import sort_contour_size
from workshop.utility import point_distance
from workshop.utility import closest_n

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class Crop(object):
    """Basic crop class"""
    def __init__(self):
        self.corners = []
    
    @property
    def contour(self):
        """Returns corners as a cv2 contour"""
        return np.array([[pt] for pt in self.corners], dtype='int')
    
    @property
    def to_tuple(self):
        """Returns corners list of tuples"""
        stack = np.vstack(self.corners).squeeze().tolist()
        return[(x,y) for x,y in stack]
    
    def save_crop(self, fp):
        """Save crop settings to external file"""
        data = np.vstack(self.corners).squeeze().tolist()
        json.dump(data, open(fp, mode='w'))

    def load_crop(self, fp):
        """Load crop settings from external file"""
        try: data = json.load(open(fp, mode='r'))
        except: return
        assert isinstance(data, list)
        self.corners = np.array(data, dtype='int')
    
    def mask_image(self, img):
        """Masks a given image by crop points"""
        #GET DIMENSIONS OF SOURCE IMAGE
        h,w = img.shape[:2]
        #CREATE BLANK MASK AT SAME DIMENSION
        mask = np.zeros((h,w,1), dtype=np.uint8)
        #DRAW MASK SHAPE FROM POINTS
        cv2.drawContours(mask, [self.contour], -1, (255), -1)
        return cv2.bitwise_and(img, img, mask=mask)
    
    def crop_image(self, img, output_dim=None):
        points = self.corners.astype('float32')
        stack = np.vstack(points).squeeze().tolist()
        pts = [(x,y) for x,y in stack]
        return corner_transform(img, pts, output_dim)
    
    def preview(self, img):
        preview_img = img.copy()
        label_contour(preview_img, self.contour, 0)
        for pt in self.to_tuple:
            cv2.circle(preview_img, pt, 5, (0, 0, 255), -1)
        return preview_img
    


class MarkerCrop(Crop):
    """Crop to tape markers of known color"""

    def __init__(self, min_color=[0,0,0], max_color=[255,255,255], marker_min=200, marker_max=2000, marker_sides=4):
        self.color = ColorRange(min_color, max_color)
        self.calibrated = False
        self.marker_min=marker_min
        self.marker_max=marker_max
        self.marker_sides = marker_sides
        super(MarkerCrop, self).__init__()

    def save_settings(self, fp=None):
        """Save settings to external configuration file"""
        data = {
            "color": self.color.save_settings(),
            "marker_sides": self.marker_sides,
            "marker_min": self.marker_min,
            "marker_max": self.marker_max
        }
        if fp: json.dump(data, open(fp, mode='w'))
        else: return data

    def load_settings(self, data):
        """Load settings from dictionary"""
        self.color.load_settings(data["color"])
        self.shape_sides = data["marker_sides"]
        self.min_area = data["marker_min"]
        self.max_area = data["marker_max"]
    
    def load_settings_file(self, fp):
        """Load settings to external configuration file"""
        try: data = json.load(open(fp, mode='r'))
        except: return
        self.load_settings(data)

    def detect(self, img, denoise=True):
        """Detect markers in a input image"""
        # RUN CALIBRATION IF FIRST PASS
        if not self.calibrated:
            self.color.calibrate(img, "MarkerCrop Color Calibration")
        mask = Mask(self.color.detect(img))
        if denoise:
            mask.morph_open(update=True)
            mask.morph_close(update=True)
        self.corners = self.get_corner_points(mask.mask)
    
    def get_corner_points(self, mask, display=False):
        """Extract inner corners from detected markers"""
        # GET CENTER POINT OF FRAME
        h,w = mask.shape[:2]
        cx = w // 2
        cy = h // 2
        # CREATE CONTOURS FROM MASK
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        # FILTER CONTOUR SHAPES BY SIZE AND NUMBER OF SIDES
        shapes = [Shape(cnt) for cnt in contours if self.marker_min < cv2.contourArea(cnt) < self.marker_max]
        shapes = [shape for shape in shapes if shape.sides == self.marker_sides]

        # IF NOT ENOUGH CORNERS ARE FOUND...
        if len(shapes) < 4:
            print("FAILED TO FIND 4 CORNERS (FOUND: {})".format(len(shapes)))
            mask_viz = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            label_contours(mask_viz, [s.contour for s in shapes])
            cv2.imshow("DETECTED CORNERS", mask_viz)
            cv2.waitkey(0)
            cv2.destroyWindow("DETECTED CORNERS")
            return None
        # IF TOO MANY CORNERS ARE FOUND...
        if len(shapes) > 4:
            print("TOO MANY CORNERS DETECTED (FOUND: {})".format)
            print("SELECTING INNER 4 CORNERS")
            shapes = sorted(shapes, key=lambda s: point_distance((cx,cy), s.centroid), reverse=False)
            shapes = shapes[:4]
        
        #RETURN INNER CORNERS OF DETECTED CROP SHAPES
        return ordered_points([shape.closest_corner((cx,cy)) for shape in shapes])
    
    def preview(self, img):
        confirm = True
        preview_img = img.copy()
        label_contour(preview_img, self.contour)
        for pt in self.to_tuple:
            cv2.circle(preview_img, pt, 5, (0, 0, 255), -1)

        cv2.imshow("CROP PREVIEW", preview_img)
        k = cv2.waitKey(0)
        if chr(k & 255) == 'x':
            print("RE-DETECTING...")
            self.detect(img)
            confirm = False
        cv2.destroyWindow("CROP PREVIEW")
        return confirm


class LightboxCrop(Crop):
    """Crop to lightbox"""

    def __init__(self, width, height, inset=0, level=100):
        self.thresh = Threshold(level,setting=3)
        self.edge = EdgeDetector()
        self.width = width
        self.height = height
        self.inset = inset
        self.calibrated = False
        super(LightboxCrop, self).__init__()
    
    def save_settings(self, fp=None):
        """Save settings to external configuration file"""
        data = {
            "edge": self.edge.save_settings(),
            "thresh":self.thresh.save_settings(),
            "width": self.width,
            "height": self.height,
            "inset": self.inset,
        }
        if fp: json.dump(data, open(fp, mode='w'))
        else: return data

    def load_settings(self, data):
        """Load settings from dictionary"""
        self.edge.load_settings(data["edge"])
        self.thresh.load_settings(data["thresh"])
        self.width = data["width"]
        self.height = data["height"]
        self.inset = data["inset"]
    
    def load_settings_file(self, fp):
        """Load settings to external configuration file"""
        try: data = json.load(open(fp, mode='r'))
        except: return
        self.load_settings(data)

    def detect(self, img, denoise=True):
        """Detect markers in a input image"""
        # RUN CALIBRATION IF FIRST PASS
        if not self.calibrated:
            self.thresh.calibrate(img, "Lightbox Crop Calibration")
        mask = Mask(self.thresh.detect(img))
        if denoise:
            mask.morph_open(update=True)
            mask.morph_close(update=True)
        self.corners = self.get_corner_points(mask.mask)
    
    def detect_edges(self, img, denoise=True):
        if not self.calibrated:
            self.edge.calibrate(img, "Lightbox Crop Calibration")
        edges = self.edge.detect(img)
        self.corners = self.get_corner_points(edges)
    
    def get_corner_points(self, mask, display=False):
        """Extract inner corners from detected markers"""
        # CREATE CONTOURS FROM MASK
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sort_contour_size(contours)
        # SELECT CONTOUR
        i = 0
        while True:
            mask_viz = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            label_contour(mask_viz, contours[i], i)
            cv2.imshow("SELECT CONTOUR", mask_viz)
            k = cv2.waitKeyEx(0)
            if k in (12,32):
                break
            if k == 2424832:
                """Left Arrow -- Decrement"""
                i = (i-1) % len(contours)
            if k == 2555904:
                """Right Arrow - Increment"""
                i = (i+1) % len(contours)
        cv2.destroyWindow("SELECT CONTOUR")
        cnt = contours[i]
        perimeter = cv2.arcLength(cnt, True)
        box = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        box = np.array([pt[0] for pt in box], dtype='int')
        box = ordered_points(box)
        box = np.float32(box)
        # cv2.cornerSubPix(mask*255, box, (11,11), (-1,-1), CRITERIA)
        # cv2.drawContours(mask_viz, [box.astype("int")], -1, (0,0,255), 2)
        # cv2.imshow("SELECTED CONTOUR", mask_viz)
        # cv2.waitKey(0)
        # cv2.destroyWindow("SELECTED CONTOUR")
        
        return ordered_points([pt for pt in box])

