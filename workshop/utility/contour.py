import cv2

def contour_center(contour):
	M = cv2.moments(contour)
	if M["m00"] == 0:
		return None 
	cx = int(M["m10"] / M["m00"])
	cy = int(M["m01"] / M["m00"])
	return cx, cy

def sort_contour(contours):
    return sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse=False)

def filter_contours(contours, min_size):
	return [cnt for cnt in contours if cv2.contourArea(cnt) > min_size]

def label_contour(img, contour, index, color=(0,255,0), thick=2):
	centroid = contour_center(contour)
	if centroid is None:
		return
	cx,cy = centroid
	
	cv2.drawContours(img, [contour], -1, color, thick)
	cv2.putText(img, str(index), (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def label_contours(img, contours):
	anno = img.copy()
	for i, cnt in enumerate(contours):
		label_contour(anno, cnt, i)
	return anno