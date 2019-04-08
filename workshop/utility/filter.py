from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

# - - - - -
# IMAGE CORRECTION FILTERS
# - - - - -
class CLAHEfilter:
    def __init__(self,clip=2.0,grid=8):
        self.clip = clip # CLIP LIMIT
        self.grid = grid # KERNEL SIZE
        self.filter = self.init_filter()

    def init_filter(self):
        return cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.grid,self.grid))

    def bw_filter(self,image,alpha=1.0):
        grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        output = self.filter.apply(grey)
        if alpha < 1.0:
            output = cv2.addWeighted(output,alpha,grey,1.0-alpha,0)
        return output

    def lab_filter(self,image,alpha=1.0):
        lab = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
        l,a,b = cv2.split(lab)
        l = self.filter.apply(l)
        output = cv2.merge((l,a,b))
        if alpha < 1.0:
            output = cv2.addWeighted(output,alpha,lab,1.0-alpha,0)
        return cv2.cvtColor(output, cv2.COLOR_Lab2BGR)

    def yuv_filter(self,image,alpha=1.0):
        yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        y,u,v = cv2.split(yuv)
        y = self.filter.apply(y)
        output = cv2.merge((y,u,v))
        if alpha < 1.0:
            output = cv2.addWeighted(output,alpha,yuv,1.0-alpha,0)
        return cv2.cvtColor(output, cv2.COLOR_YUV2BGR)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def screen(img_a, img_b=None):
    if img_b is None:
        img_b = img_a
    a = img_a.astype(float)/255
    b = img_b.astype(float)/255
    return np.uint8((1 - (1-a)*(1-b))*255)
