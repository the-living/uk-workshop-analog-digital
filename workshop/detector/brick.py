from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
import json

from math import fabs

from collections import namedtuple

from workshop.detector import Detector
from workshop.objects import Brick

from workshop.utility import contour_to_tuple

from workshop.utility import filter_contours
from workshop.utility import sort_contour
from workshop.utility import label_contours
from workshop.utility import closest_n
from workshop.utility import contour_to_tuple

from workshop.utility import ordered_points
from workshop.utility import corner_transform
from workshop.utility import rotate_img

class BrickDetector(Detector):
    """Brick detection class"""

    def __init__(self, mode=0, min_size=20000):
        super(BrickDetector, self).__init__(mode=mode, min_size=min_size)
        
        self.brick = None
        self.blob_detector = self.init_detector(mode=1)
        self.blob_calibrated = False

    @property
    def holes(self):
        return self.features
    
    def dump(self, fp):
        data = self.brick.dump()
        json.dump(data, open(fp, mode='w'))
    
    def get_holes(self, img, draw=True):
        h,w = img.shape[:2]
        brick_mask = np.zeros((h,w,1), dtype=np.uint8)
        brick_mask = cv2.drawContours(brick_mask, [self.bound.astype('int')], -1, (255), -1)
        kernel = np.ones((5,5),np.uint8)
        brick_mask = cv2.erode(brick_mask, kernel, iterations = 2)
        masked_img = cv2.bitwise_and(img, img, mask=brick_mask)
        
        if not self.blob_calibrated:
            self.calibrate(masked_img, self.blob_detector, self.blob_calibrated)
        mask = self.blob_detector.detect(masked_img)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, 100)

        holes = []
        for c in cnts:
            (x,y),r = cv2.minEnclosingCircle(c)
            holes.append((x,y,r,c))
        self.features = holes

        if draw:
            cv2.drawContours(masked_img, cnts, -1, (0,0,255), 3)
            for hole in self.holes:
                cv2.circle(masked_img, (int(x),int(y)), 5, (0,255,0), -1)
                cv2.circle(masked_img, (int(x), int(y)), int(r), (255,255,0), 1)
        return masked_img

    def make_brick(self):
        (x,y),(l,w),r = self.rect
        cnt = contour_to_tuple(self.bound)
        self.brick = Brick(x,y,l,w,r,cnt,self.holes)
    
    def update_brick(self):
        (x,y),(l,w), r = self.rect
        self.brick.depth = min(l,w)
        




        # box = cv2.minAreaRect(cnt)
        # (bx,by), (bw,bh), br = box
        # box = cv2.boxPoints(box)
        # box = ordered_points(box)

        # if fabs(br) > 45:
        #     bw,bh = bh,bw
        #     br += 90
        # brick_face = corner_transform(img,box,(int(bw),int(bh)))
            
        # if not self.blob_calibrated:
        #     self.calibrate(brick_face, self.blob_detector, self.blob_calibrated)
        # holes = self.blob_detector.detect(brick_face)
        # edge_mask = np.zeros(holes.shape, dtype=np.uint8)
        # h,w = edge_mask.shape[:2]
        # inset = 10
        # cv2.rectangle(edge_mask, (inset, inset), (w-inset, h-inset), (255), -1)
        # holes = cv2.bitwise_and(holes, edge_mask)

        # # cv2.drawContours(annotated, [box.astype("int")], -1, (255,0,0), 2)


        
        # # annotated = label_contours(annotated, cnts)
        # # self.bricks = []

        # # for c in cnts:
        # #     if cv2.contourArea(c) < self.min_size:
        # #         continue
            
        # #     box = cv2.minAreaRect(c)
        # #     (bx,by), (bw,bh), br = box
        # #     box = cv2.boxPoints(box)
            
        # #     box = ordered_points(box)

        # #     if draw:
        # #         cv2.drawContours(annotated, [box.astype("int")], -1, (0,255,0), 2)
        # #         for x,y in box:
        # #             cv2.circle(annotated, (x,y), 5, (0, 0, 255), -1)
            
        # #     if fabs(br) > 45:
        # #         bw,bh = bh,bw
        # #         br += 90
        # #     brick_face = corner_transform(img,box,(int(bw),int(bh)))
            
        # #     if not self.blob_calibrated:
        # #         self.calibrate(brick_face, self.blob_detector, self.blob_calibrated)
        # #     holes = self.blob_detector.detect(brick_face)
        # #     edge_mask = np.zeros(holes.shape, dtype=np.uint8)
        # #     h,w = edge_mask.shape[:2]
        # #     inset = 10
        # #     cv2.rectangle(edge_mask, (inset, inset), (w-inset, h-inset), (255), -1)
        # #     holes = cv2.bitwise_and(holes, edge_mask)

            
        # #     if show:
        # #         cv2.imshow("holes", holes)
        # #         cv2.waitKey(0)

        # #     hole_contours = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # #     hole_contours = sort_contour(hole_contours)
        # #     brick_face = label_contours(brick_face, hole_contours)
        # #     hole_data = []
        # #     for cnt in hole_contours:
        # #         (hx,hy),radius = cv2.minEnclosingCircle(cnt)
        # #         center = (int(hx),int(hy))
        # #         radius = int(radius)
        # #         cv2.circle(brick_face,center,radius,(0,0,255),2)
        # #         hole_data.append( (hx,hy,radius,cnt))
        # #     if show:
        # #         cv2.imshow("holes", brick_face)
        # #         cv2.waitKey(0)
        # #         cv2.destroyWindow("holes")

        # #     self.bricks.append( Brick(bx, by, br, bw, bh, box, hole_data))
        
        # # if verbose:
        # #     for brick in self.bricks:
        # #         print(brick)
        
        
        # # if show and pause:
        # #     cv2.waitKey(0)
        
        # # if show:
        # #     cv2.destroyWindow('img')
        
        # return annotated
    
    def detect_many(self, img, invert=False, draw=True, show=False, pause=False, verbose=False):
        
        if verbose:
            print("DETECTING...")
        if not self.calibrated:
            self.calibrate(img)
        img = self.prep_image(img)

        mask = self.detector.detect(img)
        if invert:
            mask = cv2.bitwise_not(mask)

        annotated = img.copy()

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = filter_contours(cnts, self.min_size)
        cnts = sort_contour(cnts)
        
        self.bricks = []

        for c in cnts:
            if cv2.contourArea(c) < self.min_size:
                continue
            
            box = cv2.minAreaRect(c)
            (bx,by), (bw,bh), br = box
            box = cv2.boxPoints(box)
            
            box = ordered_points(box)

            if draw:
                cv2.drawContours(annotated, [box.astype("int")], -1, (0,255,0), 2)
                for x,y in box:
                    cv2.circle(annotated, (x,y), 5, (0, 0, 255), -1)
            
            if fabs(br) > 45:
                bw,bh = bh,bw
                br += 90
            brick_face = corner_transform(img,box,(int(bw),int(bh)))
            
            if not self.blob_calibrated:
                self.calibrate(brick_face, self.blob_detector, self.blob_calibrated)
            holes = self.blob_detector.detect(brick_face)
            edge_mask = np.zeros(holes.shape, dtype=np.uint8)
            h,w = edge_mask.shape[:2]
            inset = 10
            cv2.rectangle(edge_mask, (inset, inset), (w-inset, h-inset), (255), -1)
            holes = cv2.bitwise_and(holes, edge_mask)

            
            if show:
                cv2.imshow("holes", holes)
                cv2.waitKey(0)

            hole_contours = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            hole_contours = sort_contour(hole_contours)
            brick_face = label_contours(brick_face, hole_contours)
            hole_data = []
            for cnt in hole_contours:
                (hx,hy),radius = cv2.minEnclosingCircle(cnt)
                center = (int(hx),int(hy))
                radius = int(radius)
                cv2.circle(brick_face,center,radius,(0,0,255),2)
                hole_data.append( (hx,hy,radius,cnt))
            if show:
                cv2.imshow("holes", brick_face)
                cv2.waitKey(0)
                cv2.destroyWindow("holes")

            self.bricks.append( Brick(bx, by, br, bw, bh, box, hole_data))
        
        if verbose:
            for brick in self.bricks:
                print(brick)
        
        
        if show and pause:
            cv2.waitKey(0)
        
        if show:
            cv2.destroyWindow('img')
        
        return annotated

        

    

    
