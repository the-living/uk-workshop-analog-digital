import cv2

class EdgeDetector(object):

    def __init__(self, lower=50, upper=100):
        self.lower = lower
        self.upper = upper
    
    def detect(self, img, open=True):
        edges = cv2.Canny(img, self.lower, self.upper)
        if open:
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)
        return edges
    
    def calibrate(self, img):
        while True:
            # DRAW SETTING ON IMAGE
            calib = cv2.cvtColor(self.detect(img), cv2.COLOR_GRAY2BGR)
            label = "CANNY {} : {}".format(self.lower, self.upper)
            cv2.putText(calib, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            # SHOW CURRENT EGDE DETECTION
            cv2.imshow("CANNY Calibration", calib)

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
        cv2.destroyWindow("CANNY Calibration")
