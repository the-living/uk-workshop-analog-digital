from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os

from math import fabs

import cv2
import numpy as np

from workshop.detector import Detector
from workshop.objects import Wood
from workshop.detector import extract_grain

from workshop.utility import sort_contour
from workshop.utility import filter_contours
from workshop.utility import label_contours
from workshop.utility import ordered_points

from workshop.utility import corner_transform

class WoodDetector(Detector):
    """2x4 Off-cut Detector Class"""
    def __init__(self, mode=0, min_size=20000):
        super(WoodDetector, self).__init__(mode, min_size)
        self.grain_detector = None
        self.grain_thresh = self.init_detector(mode=1)
        self.grain_calibrated = False

        

        self.cuts = []

    def dump(self, fp):
        data = [b.dump() for b in self.cuts]
        json.dump(data, open(fp, mode='w'))
    
    def detect(self, img_edge, img):
        print("DETECTING FROM BACKLIT...")
        if not self.calibrated:
            self.calibrate(img_edge)
        # detect = self.prep_image(img)

        h,w = img.shape[:2]    
        

        mask = self.detector.detect(img_edge) 
        mask = cv2.bitwise_not(mask)

        annotated = img.copy()

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, self.min_size)
        cnts = sort_contour(cnts)

        epsilon = lambda c: 0.004 * cv2.arcLength(c, True)
        approximate = lambda c: cv2.approxPolyDP(c, epsilon(c), True)
        approx = [approximate(c) for c in cnts]

        min_rect = lambda x: cv2.minAreaRect(x)
        min_box = lambda x: ordered_points(cv2.boxPoints(x))
        min_boxes = [min_rect(c) for c in approx]
        boxes = [min_box(c) for c in min_boxes]
    
        board_mask = np.zeros((h,w,1), dtype=np.uint8)
        cv2.drawContours(board_mask, approx, -1, (255), -1)

        cv2.drawContours(annotated, boxes, -1, (0,255,0), 2)
        cv2.drawContours(annotated, approx, -1, (255,0,0), 2)

        for bnd, box in zip(approx, min_boxes):
            (cx,cy),(cw,ch),r = box
            self.cuts.append(Wood(cx, cy, cw, ch, r, bnd))
        
        for box, corners in zip(min_boxes, boxes):
            (cx,cy),(cw,ch),r = box
            if fabs(r) > 45:
                    cw,ch = ch,cw
                    br += 90
            wood_face = corner_transform(img, corners, (int(cw),int(ch)))
            wood_mask = corner_transform(board_mask, corners, (int(cw), int(ch)))

            grain = extract_grain(wood_face, horizontal=True)
            grain = np.array(grain, dtype=np.uint8)
            if not self.grain_calibrated:
                self.grain_thresh.calibrate(grain)
            grain = self.grain_thresh.detect(grain)
            grain = cv2.bitwise_not(grain)
            # grain = cv2.morphologyEx(grain, cv2.MORPH_OPEN, (3,3), iterations=1)

            # print(grain.shape, wood_mask.shape)
            grain = cv2.bitwise_and(grain, grain, mask=wood_mask)
            grain_cnts = cv2.findContours(grain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            grain_cnts = filter_contours(grain_cnts, 200)
            wood_face_grain = wood_face.copy()
            wood_face_grain = label_contours(wood_face_grain, grain_cnts)

            
            # if not self.blob_calibrated:
            #     self.calibrate(brick_face, self.blob_detector)
            # holes = self.blob_detector.detect(brick_face)
            # edge_mask = np.zeros(holes.shape, dtype=np.uint8)
            # h,w = edge_mask.shape[:2]
            # inset = 10
            # cv2.rectangle(edge_mask, (inset, inset), (w-inset, h-inset), (255), -1)
            # holes = cv2.bitwise_and(holes, edge_mask)

            # cv2.imshow("face", wood_face)
            # cv2.imshow("grain", grain)
            # cv2.imshow("grain_cnt", wood_face_grain)
            # cv2.waitKey(0)

        # for c in cnts:
        #     if cv2.contourArea(c) < self.min_size:
        #         continue
            
        #     perimeter = cv2.arcLength(c, True)
            
        #     approx = cv2.approxPolyDP(c, 0.003 * perimeter, True)
            
        #     box = cv2.minAreaRect(c)
        #     (bx,by), (bw,bh), br = box
        #     box = cv2.boxPoints(box)
            
        #     box = ordered_points(box)

        #     cv2.drawContours(annotated, [box.astype("int")], -1, (0,255,0), 2)
        #     for x,y in box:
        #         cv2.circle(annotated, (x,y), 5, (0, 0, 255), -1)
        #     cv2.drawContours(annotated, [approx], -1, (255,0,0), 2)
        #     for pt in approx:
        #         x,y = pt[0]
        #         cv2.circle(annotated, (x,y), 2, (255, 0, 255), -1)
            
        
        
        cv2.destroyAllWindows()
        return annotated

