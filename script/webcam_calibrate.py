import cv2
import numpy as np

import os
import time

from workshop.utility import CircleGrid
from workshop.utility import Chessboard

w,h = (1920, 1080)

feed = cv2.VideoCapture(1)
feed.set(3,h)
feed.set(4,w)

out_dir = r"__IMG\capture"
calib_dir = r"settings\camera_calibration"


cb = CircleGrid()
try:
    cb.load_calibration(os.path.join(calib_dir,"calibration.json"))
except:
    print("CIRCLE GRID CALIBRATION")
    cb.calibrate(feed, frames=100)
    cb.save_calibration(os.path.join(calib_dir,"calibration.json"))

# print("CHESSBOARD CALIBRATION")
# cb = Chessboard(x=7,y=7)
# cb.calibrate(feed, frames=60)
# cb.save_calibration("chess_calibration.json")

while True:
    ret, frame = feed.read()

    frame_undistorted = cb.undistort(frame, crop=True)

    cv2.imshow('original', frame)
    cv2.imshow('undistorted', frame_undistorted)

    key = cv2.waitKey(1)
    if key == 27:
        print("EXITING...")
        break
    if chr(key % 256) == 'x':
        label = "calibrate_{}.jpg".format(time.time())
        label_undistort = "calibrate_{}.undistorted.jpg".format(time.time())
        print("SAVING... {}".format(label))
        cv2.imwrite(os.path.join(out_dir, label), frame)
        cv2.imwrite(os.path.join(out_dir, label_undistort), frame_undistorted)
    if chr(key % 256) == 'c':
        print("CIRCLE GRID CALIBRATION")
        cv2.destroyAllWindows()
        cb = CircleGrid()
        cb.calibrate(feed, frames=100)
        cb.save_calibration(os.path.join(calib_dir,"calibration.json"))


feed.release()
cv2.destroyAllWindows()