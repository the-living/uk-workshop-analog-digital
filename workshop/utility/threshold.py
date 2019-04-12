import cv2
import json

modes = ["Normal", "AdaptiveMean", "AdaptiveGaussian", "OTSU"]

class Threshold(object):

    def __init__(self, level=50, output=255, setting=0):
        self.level = level
        self.output = output
        self.mode = int(setting % len(modes))
        self.block = 11
        self.blur = 5

    def detect(self, img):
        gray = img.copy()
        if len(gray.shape)>2 and gray.shape[2] > 1:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.mode == 3:
            # OTSU
            blurred = cv2.GaussianBlur(gray, (self.blur,self.blur), 0)
            return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if self.mode == 2:
            # ADAPTIVE GAUSSIAN
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block, 2)
        if self.mode == 1:
            # ADAPTIVE MEAN
            return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,self.block,2)
        # NORMAL
        return cv2.threshold(gray, self.level, 255, cv2.THRESH_BINARY)[1]

    
    def calibrate(self, img, window_label="THRESHOLD Calibration"):
        while True:
            # DRAW SETTING ON IMAGE
            calib = cv2.cvtColor(self.detect(img), cv2.COLOR_GRAY2BGR)
            label = modes[self.mode]
            if self.mode == 0:
                label += " : {}".format(self.level)
            if self.mode in [1,2]:
                label += " : {}".format(self.block)
            if self.mode == 3:
                label += " : {}".format(self.blur)
            cv2.putText(calib, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            # SHOW CURRENT EGDE DETECTION
            cv2.imshow(window_label, calib)

            # WAIT FOR KEYBOARD INPUT
            k = cv2.waitKeyEx(0)
            if k in (27, 13, 32):
                """ESC/SPACE/ENTER - Exit calibration"""
                break
            if k == 2490368:
                """Up Arrow -- Increase level"""
                if self.mode == 0:
                    self.level = max(0, min(255, self.level + 5))
                if self.mode in [1,2]:
                    self.block = self.block + 2
                if self.mode == 3:
                    self.blur = self.blur + 2
            if k == 2621440:
                """Down Arrow -- Decrease level"""
                if self.mode == 0:
                    self.level = max(0, min(255, self.level - 5))
                if self.mode in [1,2]:
                    self.block = max(3, self.block - 2)
                if self.mode == 3:
                    self.blur = max(3, self.blur - 2)
            if k == 2424832:
                """Left Arrow -- Change mode"""
                self.mode = (self.mode - 1) % len(modes)
            if k == 2555904:
                """Right Arrow - Increase lower"""
                self.mode = (self.mode + 1) % len(modes)
            
        # KILL CALIBRATION WINDOW
        cv2.destroyWindow(window_label)
    
    def save_settings(self):
        return {"level": self.level, "output":self.output, "mode":self.mode, "block":self.block, "blur":self.blur}
    
    def load_settings(self, data):
        self.level = data["level"]
        self.output = data["output"]
        self.mode = data["mode"]
        self.block = data["block"]
        self.blur = data["blur"]
    
    def load_settings_file(self, fp):
        data = json.load(open(fp, mode='r'))
        self.load_settings(data)
