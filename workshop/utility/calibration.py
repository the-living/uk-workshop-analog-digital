from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
import json

from math import floor

__all__ = ["Chessboard", "CircleGrid"]

class Calibration(object):
    """Generic calibration class"""
    def __init__(self, objpt=None, calibration={}):
        self.objpt = objpt
        self.calibration = {}
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    @property
    def camera_matrix(self):
        return self.calibration["camera_matrix"]
    
    @camera_matrix.setter
    def camera_matrix(self, value):
        self.calibration["camera_matrix"] = value
    
    @property
    def dist_coeff(self):
        return self.calibration["dist_coeff"]
    
    @dist_coeff.setter
    def dist_coeff(self, value):
        self.calibration["dist_coeff"] = value


    def save_calibration(self, fp=None):
        if fp is None:
            fp = "calibration.json"
        data = {
            "camera_matrix": np.asarray(self.camera_matrix).tolist(),
            "dist_coeff": np.asarray(self.dist_coeff).tolist()
        }
        with open(fp, mode='w') as f:
            json.dump(data, f, separators=(',',': '), indent=4)
        f.close()
    
    def load_calibration(self, fp):
        with open(fp, mode='r') as f:
            self.calibration = json.load(f)
            self.calibration["camera_matrix"] = np.array(self.calibration["camera_matrix"])
            self.calibration["dist_coeff"] = np.array(self.calibration["dist_coeff"])
        f.close()

    def undistort(self, img, crop=True, reverse=False):
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeff, (w,h), 1, (w,h))
        
        dst = cv2.undistort(
            img, self.camera_matrix, self.dist_coeff, None, new_camera_matrix)

        x,y,w,h = roi
        if crop:
            dst = dst[y:y+h, x:x+w]
        return dst

class Chessboard(Calibration):
    """Chessboard camera calibration module"""
    def __init__(self, x=7, y=7, spacing=30):
        self.x = x
        self.y = y
        self.spacing = spacing
        super(Chessboard, self).__init__(self.__gen_grid_pts())
    
    def __gen_grid_pts(self):
        count = int(self.x * self.y)
        objp = np.zeros((count, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.y, 0:self.x].T.reshape(-1,2)
        objp *= self.spacing
        return objp
    
    def calibrate(self, feed, frames=20, display=True):
        objpoints = [] #3D Points in RWC
        imgpoints = [] #2D Points in Image Plane

        found = 0
        kill = 0

        while found < frames:
            if kill > (frames * 5):
                print("TIMEOUT...")
                break
            kill += 1

            # CAPTURE FRAME FROM FEED
            ret, img = feed.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.x, self.y), None)

            if ret == True:
                objpoints.append(self.objpt)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners2)

                img = cv2.drawChessboardCorners(img, (self.x, self.y), corners2, ret)
                found += 1

            if display:  
                cv2.imshow("calibration", img)
            cv2.waitKey(2)
        
        cv2.destroyAllWindows()
        
        ret, self.camera_matrix, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)


class CircleGrid(Calibration):
    """
    Camera calibration module using circle grid
    https://nerian.com/support/resources/patterns/pattern-letter.pdf
    """

    def __init__(self, x=4, y=11, spacing=20, diameter=15, assymetric=True):
        self.x = x # rows in grid
        self.y = y # columns in grid
        self.spacing = spacing # spacing between 
        self.diameter = diameter
        self.assymetric = True
        self.blob_detector = self.__setup_blob_detector()

        super(CircleGrid, self).__init__(self.__gen_obj_pts())
        

    def __gen_obj_pts(self):
        count = int(self.x * self.y)
        pts = np.zeros((count, 3), np.float32)
        for i in range(count):
            if self.assymetric:
                x = floor(i / self.x) / 2
                y = i % self.x
                if floor(i / self.x) % 2 == 1:
                    y += 0.5
            else:
                x = floor(i / self.x)
                y = i % self.x
            x *= self.spacing
            y *= self.spacing
            pts[i] = (x, y, 0)
        return pts
    
    def __setup_blob_detector(self):
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minThreshold = 8
        blob_params.maxThreshold = 255
        blob_params.filterByArea = True
        blob_params.minArea = 25
        blob_params.maxArea = 2500
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.1
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.87
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.01
        return cv2.SimpleBlobDetector_create(blob_params)

    def calibrate(self, feed, frames=20, display=True):
        objpoints = [] #3D Points in RWC
        imgpoints = [] #2D Points in Image Plane

        found = 0
        kill = 0

        while found < frames:
            # if kill > (frames * 8):
            #     print("TIMEOUT...")
            #     break
            # kill += 1

            # CAPTURE FRAME FROM FEED
            ret, img = feed.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints = self.blob_detector.detect(gray)

            img_kp = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   
            # img_kp = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255)) 


            img_kp_gray = cv2.cvtColor(img_kp, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findCirclesGrid(
                img_kp_gray, (self.x, self.y), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=self.blob_detector)

            if ret == True:
                objpoints.append(self.objpt)
                
                # corners2 = cv2.cornerSubPix(img_kp_gray, corners, (11,11), (-1,-1), self.criteria)
                # imgpoints.append(corners2)
                # img_kp = cv2.drawChessboardCorners(img, (self.x, self.y), corners2, ret)

                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(img, (self.x, self.y), corners, ret)

                found += 1
                # cv2.waitKey(0)
            label = "CALIBRATIONS: {:d}".format(found)
            cv2.putText(img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            
            if display:
                cv2.imshow('blob', img_kp)
                cv2.imshow("calibration", img)
            c = cv2.waitKey(2)
            if c == 27:
                break
        
        cv2.destroyAllWindows()

        ret, self.camera_matrix, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
