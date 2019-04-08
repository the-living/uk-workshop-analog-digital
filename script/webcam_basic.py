import cv2
import numpy as np

w,h = (1920, 1080)

feed = cv2.VideoCapture(1)
feed.set(3,h)
feed.set(4,w)

while True:
    ret, frame = feed.read()

    cv2.imshow('feed', frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("EXITING...")
        break

feed.release()
cv2.destroyAllWindows()