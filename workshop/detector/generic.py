from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import json

from workshop.utility import ColorRange
from workshop.utility import Threshold
from workshop.utility import EdgeDetector

from workshop.utility import CLAHEfilter
from workshop.utility import white_balance
from workshop.utility import screen

from workshop.utility import point_distance
from workshop.utility import interpolate_point

from workshop.utility import filter_contours
from workshop.utility import closest_n


class Detector(object):
    """Generic detector class"""
    def __init__(self, mode=0, min_size=2000):
        self.min_size = min_size
        self.detect_mode = mode
        self.detector = self.init_detector()
        self.calibrated = False

        self.bound = None
        self.rect = None
        self.features = None

        self.clahe = CLAHEfilter()

    def init_detector(self, mode=None):
        """Initialize a new detector"""
        if mode is None:
            mode = self.detect_mode
        mode = mode % 3
        if mode == 0:
            return ColorRange()
        if mode == 1:
            return Threshold()
        if mode == 2:
            return EdgeDetector()
    
    def reset(self):
        self.bound = None
        self.rect = None
        self.features = None
    
    def prep_image(self, img, screen_pass=1, wb=False, clahe=True, b_kernel=11, b_sigma=25):
        image = img.copy()
        if b_kernel:
            image = cv2.bilateralFilter(image, b_kernel, b_sigma, b_sigma)
        if wb:
            image = white_balance(image)
        if clahe:
            image = self.clahe.yuv_filter(image)
        for _ in range(int(screen_pass)):
            image = screen(image)
        return image
    
    def calibrate(self, img, d=None, d_flag=None):
        """Run calibration on detector"""
        if d is None:
            d = self.detector
        if d_flag is None:
            d_flag = self.calibrated
        calib = self.prep_image(img)
        d.calibrate(calib)
        d_flag = True
    
    def save_calibration(self, fp):
        data = {
            "mode":self.detect_mode,
            "detector":self.detector.save_settings(),
            "min_size": self.min_size
            }
        json.dump(data, open(fp, mode='w'))
    
    def load_calibration(self, data):
        self.detect_mode = data["mode"]
        self.detector = self.init_detector()
        self.detector.load_settings(data["detector"])
        self.min_size = data["min_size"]
    
    def load_calibration_file(self, fp):
        data = json.load(open(fp, mode='r'))
        self.load_calibration(data)
    
    def detect_edge(self, img, invert=False, draw=True, show=False, pause=False, verbose=False):
        self.reset()

        if verbose:
            print("DETECTING...")
        if not self.calibrated:
            self.calibrate(img)
        img = self.prep_image(img)

        mask = self.detector.detect(img)
        if self.detect_mode == 1:
            mask = cv2.bitwise_not(mask)

        annotated = img.copy()

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, self.min_size)
        # cnts = sort_contour(cnts)

        h,w = img.shape[:2]
        cnt = closest_n(cnts, w//2, h//2)[0]

        self.bound = cnt
        if draw:
            cv2.drawContours(annotated, [self.bound], -1, (0,255,0), 2)
        return annotated
    
    def get_box(self, img=None):
        if self.bound is None:
            return
        self.rect = cv2.minAreaRect(self.bound)
        if img is not None:
            annotated = img.copy()
            box = cv2.boxPoints(self.rect)
            cv2.drawContours(annotated, [box.astype('int')], -1, (255,0,0), 1)
            for x,y in box:
                cv2.circle(annotated, (x,y), 5, (0,0,255), -1)
            return annotated