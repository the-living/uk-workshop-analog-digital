import cv2
import numpy as np



from workshop import utility
from workshop import detector

frame_w = 1920
frame_h = 1080

margins = (250,100,200,200)
crop_w = 1220
crop_h = 915

cb = utility.CircleGrid()
cb.load_calibration(r"settings\camera_calibration\calibration.json")

crop = utility.MarkerCrop((100, 50, 0),(120, 200, 255))

brick = detector.BrickDetector()

feed = cv2.VideoCapture(1)
feed.set(3,frame_h)
feed.set(4,frame_w)

detection = False
init = True

while True:
    ret, image = feed.read()

    # CROP IMAGE MARGINS
    h,w = image.shape[:2]
    top, bottom, left, right = margins
    image = image[top:h-bottom, left:w-right]


    # UNDISTORT IMAGE
    image = cb.undistort(image, crop=True)

    if init:
        cv2.imshow("UNDISTORTED", image)
        k = cv2.waitKey(0)
        if k == 27:
            print("EXITING...")
            break
        cv2.destroyAllWindows()
    
    if init:
        ret = False
        while not ret:
            ret = crop.preview(image)
    cropped = crop.crop_image(image, (crop_w, crop_h))

    # cv2.imshow('RAW', cropped)

    frame = brick.detect(cropped)
    cv2.imshow('IMG', frame)
    
    init = False
    c = cv2.waitKey(1000)
    if c == 27:
        break
    

cv2.destroyAllWindows()