import cv2
import numpy as np

while True:
    cv2.imshow("img", np.zeros((10,10,1), np.uint8))
    k = cv2.waitKeyEx(0)
    if k == 27:
        break
    print("{} : {}".format(k, chr(k % 255)))