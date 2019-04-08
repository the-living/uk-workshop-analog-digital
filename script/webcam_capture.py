import cv2
import numpy as np

import os
import time

w,h = (1920, 1080)

feed = cv2.VideoCapture(1)
feed.set(3,h)
feed.set(4,w)

out_dir = "capture"


cv2.namedWindow('feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = feed.read()

    cv2.imshow('feed', frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("EXITING...")
        break
    if chr(key % 256) == 'x':
        label = "cap_{}.jpg".format(time.time())
        print("SAVING... {}".format(label))
        cv2.imwrite(os.path.join(out_dir, label), frame)

feed.release()
cv2.destroyAllWindows()