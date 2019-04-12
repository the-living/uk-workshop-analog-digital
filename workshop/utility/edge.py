import cv2
import json

class EdgeDetector(object):

    def __init__(self, lower=50, upper=100):
        self.lower = lower
        self.upper = upper
    
    def detect(self, img, open=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(img, (7,7), 0)
        edges = cv2.Canny(img, self.lower, self.upper)
        if open:
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)
        return edges
    
    def calibrate(self, img, window_label="EdgeDetector Calibration"):
        while True:
            # DRAW SETTING ON IMAGE
            calib = cv2.cvtColor(self.detect(img), cv2.COLOR_GRAY2BGR)
            label = "CANNY {} : {}".format(self.lower, self.upper)
            cv2.putText(calib, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            # SHOW CURRENT EGDE DETECTION
            cv2.imshow(window_label, calib)

            # WAIT FOR KEYBOARD INPUT
            k = cv2.waitKeyEx(0)
            if k in (27, 13, 32):
                """ESC/SPACE/ENTER - Exit calibration"""
                break
            if k == 2490368:
                """Up Arrow -- Increase upper"""
                self.upper = max(0, min(255, self.upper + 5))
            if k == 2621440:
                """Down Arrow -- Decrease upper"""
                self.upper = max(0, min(255, self.upper - 5))
            if k == 2424832:
                """Left Arrow -- Decrease lower"""
                self.lower = max(0, min(255, self.lower - 5))
            if k == 2555904:
                """Right Arrow - Increase lower"""
                self.lower = max(0, min(255, self.lower + 5))
            
        # KILL CALIBRATION WINDOW
        cv2.destroyWindow(window_label)
    
    def save_settings(self):
        return {"lower": self.lower, "upper": self.upper}
        
    def load_settings(self, data):    
        self.lower = data["lower"]
        self.upper = data["upper"]
    
    def load_settings_file(self, fp):
        data = json.load(open(fp, mode='r'))
        self.load_settings(data)
