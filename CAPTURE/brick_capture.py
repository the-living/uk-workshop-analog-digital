import cv2
import numpy as np

import os
import json
import glob

from workshop import utility

from workshop.utility import LightboxCrop
from workshop.detector import BrickDetector
from workshop.utility import Calibration
from workshop.utility import white_balance

PROJECT_DIR = r""

CAM_CALIBRATION = os.path.join(PROJECT_DIR, r"settings\camera.json")
LB_CALIBRATION = os.path.join(PROJECT_DIR, r"settings\lightbox_settings.json")
BRICK_CALIBRATION = os.path.join(PROJECT_DIR, r"settings\detector_settings.json")

BRICK_OUTPUT = os.path.join(PROJECT_DIR, r"output\bricks")

def draw_label(img, text_lines):
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (20,20 + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


w,h = (1920, 1080)
feed = cv2.VideoCapture(1)
feed.set(3,w)
feed.set(4,h)

lb_w, lb_h = (330,430)

cb = utility.CircleGrid()
try:
    cb.load_calibration(CAM_CALIBRATION)
except:
    cb.calibrate(feed, frames=60)
while True:
    ret, frame = feed.read()
    frame = cb.undistort(frame, crop=False)

    msg1 = "CAMERA UNDISTORT"
    msg2 = "[SPACE]: CONFIRM [C]: RE-CALIBRATE"
    msg = [msg1, msg2]
    draw_label(frame, msg)

    cv2.imshow('undistorted', frame)
    k = cv2.waitKey(0)
    if k == 32:
        break
    if chr(k % 255) == 'c':
        cb.calibrate(feed, frames=60)
cb.save_calibration(CAM_CALIBRATION)
cv2.destroyAllWindows()

lightbox = LightboxCrop(width=lb_w, height=lb_h, inset=5, level=100)
if os.path.exists(LB_CALIBRATION):
    lightbox.load_settings_file(LB_CALIBRATION)

detector = BrickDetector(mode=1)
detector.detector.blur = 11


def run(calibrated=False):
    if not calibrated:
        print("\n\nINITIAL CALIBRATION\n")
        print("TURN ON LIGHTBOX WITH NO OBJECTS ON IT")
        print("WHEN READY, PRESS [SPACE BAR] TO CONTINUE")
        while True:
            ret, frame = feed.read()
            print(frame.shape)
            frame = white_balance(frame)
            print(frame.shape)
            frame = cb.undistort(frame)
            print(frame.shape)

            msg1 = "TURN ON LIGHT BOX"
            msg2 = "[SPACE]: CAPTURE [ESC]: QUIT"
            msg = [msg1, msg2]
            draw_label(frame, msg)

            cv2.imshow('DETECTION', frame)
            k = cv2.waitKey(10)
            if k == 27:
                return False
            if k == 32:
                break

        print("\n\nDETECTING LIGHTBOX AREA...")
        print("PRESS [SPACE BAR] TO CONTINUE")
        print("PRESS [X] TO RE-DETECT")
        # lightbox.detect(frame)
        lightbox.detect_edges(frame)
        while True:
            ret, frame = feed.read()
            frame = cb.undistort(frame)

            frame = lightbox.preview(frame)

            msg1 = "DETECTION AREA"
            msg2 = "[SPACE]: CONFIRM [X]: RE-CALIBRATE [ESC]: QUIT"
            msg = [msg1, msg2]
            draw_label(frame, msg)

            cv2.imshow('DETECTION', frame)
            k = cv2.waitKey(10)
            if k == 27:
                return False
            if k == 32:
                break
            if chr(k & 255) == "x":
                # lightbox.detect(frame)
                lightbox.detect_edges(frame)
        
    print("\n\nOBJECT DETECTION\n")
    print("PLACE OBJECT ON CAPTURE AREA")
    print("PRESS [SPACE BAR] TO CAPTURE EDGES")
    while True:
        ret, frame = feed.read()
        frame = cb.undistort(frame)

        cropped = lightbox.crop_image(frame, (840, 640))

        msg1 = "PLACE OBJECT ON DETECTION SURFACE"
        msg2 = "[SPACE]: CONFIRM [ESC]: QUIT"
        msg = [msg1, msg2]
        draw_label(cropped, msg)

        cv2.imshow('DETECTION', cropped)
        k = cv2.waitKey(10)
        if k == 27:
            return False
        if k == 32:
            break
    
    print("\n\nOBJECT RECORDING\n")
    print("PRESS [SPACE BAR] TO CAPTURE HOLES")
    while True:
        ret, frame = feed.read()
        frame = cb.undistort(frame)

        cropped = lightbox.crop_image(frame, (840, 640))

        detection = detector.detect_edge(cropped)
        detection = detector.get_box(detection)

        msg1 = "DETECTED SHAPE"
        msg2 = "[SPACE]: CONFIRM [X]: RE-CALIBRATE [ESC]: QUIT"
        msg = [msg1, msg2]
        draw_label(detection, msg)

        cv2.imshow('DETECTION', detection)
        k = cv2.waitKey(10)
        if k == 27:
            return False
        if k == 32:
            break
        if chr(k & 255) == "x":
                detector.calibrate(cropped)
        detector.calibrated = True
    
    print("\n\nFEATURE CAPTURE\n")
    print("PRESS [SPACE BAR] TO RECORD HOLES")
    while True:
        ret, frame = feed.read()
        frame = cb.undistort(frame)

        cropped = lightbox.crop_image(frame, (840, 640))

        detection = detector.get_holes(cropped)

        msg1 = "DETECTED FEATURES"
        msg2 = "[SPACE]: CONFIRM [X]: RE-CALIBRATE [ESC]: QUIT"
        msg = [msg1, msg2]
        draw_label(detection, msg)

        cv2.imshow('DETECTION', detection)
        k = cv2.waitKey(10)
        if k == 27:
            return False
        if k == 32:
            break
        if chr(k & 255) == "x":
                detector.calibrate(cropped, detector.blob_detector, detector.blob_calibrated)
        detector.blob_calibrated = True
    
    print("\n\nGENERATING BRICK OBJECT\n")
    detector.make_brick()
    
    print("\n\nSECOND FACE DETECTION\n")
    print("ROTATE OBJECT ON CAPTURE AREA ONTO EDGE")
    print("PRESS [SPACE BAR] TO CAPTURE SECOND EDGES")
    while True:
        ret, frame = feed.read()
        frame = cb.undistort(frame)

        cropped = lightbox.crop_image(frame, (840, 640))

        msg1 = "ROTATE OBJECT TO CAPTURE SECOND SIDE"
        msg2 = "[SPACE]: CONFIRM [ESC]: QUIT"
        msg = [msg1, msg2]
        draw_label(cropped, msg)

        cv2.imshow('DETECTION', cropped)
        k = cv2.waitKey(10)
        if k == 27:
            return False
        if k == 32:
            break
    
    print("\n\nOBJECT RECORDING\n")
    print("PRESS [SPACE BAR] TO SAVE BRICK OBJECT")
    while True:
        ret, frame = feed.read()
        frame = cb.undistort(frame)

        cropped = lightbox.crop_image(frame, (840, 640))

        detection = detector.detect_edge(cropped)
        detection = detector.get_box(detection)

        msg1 = "DETECTED SHAPE"
        msg2 = "[SPACE]: CONFIRM [X]: RE-CALIBRATE [ESC]: QUIT"
        msg = [msg1, msg2]
        draw_label(detection, msg)

        cv2.imshow('DETECTION', detection)
        k = cv2.waitKey(10)
        if k == 27:
            return False
        if k == 32:
            break
        if chr(k & 255) == "x":
                detector.calibrate(cropped)
        detector.calibrated = True
    
    detector.update_brick()
    brick_count = len(glob.glob(os.path.join(BRICK_OUTPUT, "*.json")))
    label = os.path.join(BRICK_OUTPUT, "brick_{:04d}.json".format(brick_count))
    detector.dump(label)
    print("\nSAVED BRICK TO: {}".format(label))


    print("\n\nTAG OBJECT\n")
    print("PRINT AND APPLY LABEL TO OBJECT")
    print("PRESS [SPACE BAR] TO CONTINUE")
    print("PRESS [ESC] TO QUIT")
    frame = np.zeros(cropped.shape[:2], dtype=np.uint8)
    msg1 = "LABEL THIS BRICK"
    msg2 = "{:04}".format(brick_count)
    cv2.putText(frame, msg1, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 2, cv2.LINE_AA)
    cv2.putText(frame, msg2, (20,240), cv2.FONT_HERSHEY_DUPLEX, 4, (255), 2, cv2.LINE_AA)

    while True:
        cv2.imshow('DETECTION', frame)
        k = cv2.waitKey(0)
        if k == 27:
            return False
        if k == 32:
            break



    
    

    



    # detector = BrickDetector(mode=1)

    # cropped = detector.detect(cropped, invert=True, draw=True, show=True)
    # detector.dump(BRICK_OUTPUT)

    # cv2.imshow("LIGHTBOX", cropped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # lightbox.save_settings(LB_CALIBRATION)
    # # crop.save_crop(CROP_CORNERS)
    # detector.save_calibration(BRICK_CALIBRATION)

    return True

if __name__ == "__main__":
    first_pass = False
    running = True
    while running:
        running = run(first_pass)
        first_pass = True
    print("\n...SAVING CALIBRATION SETTINGS...\n")
    lightbox.save_settings(LB_CALIBRATION)
    detector.save_calibration(BRICK_CALIBRATION)
    print("\n...EXITING...\n")
    feed.release()
    cv2.destroyAllWindows()