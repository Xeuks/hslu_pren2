import numpy as np
import cv2
import sys


# import the necessary packages
import cv2

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
	
		approx = cv2.approxPolyDP(c,  peri, True)

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		if len(approx) == 4:
                        # compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square"
			

		# return the name of the shape
		return shape

sd = ShapeDetector()
cap = cv2.VideoCapture(0)

while(True):
        gray = cv2.imread("zielfeld.png", 0)#cap.read()[1]
      #  gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),5)
       # ratio = img.shape[0]
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        edged = cv2.Canny(gray, 200, 300)
        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] #if imutils.is_cv2() else cnts[1]

        # loop over the contours
        
        for c in cnts:
               

                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                M = cv2.moments(c)
                if M["m00"] != 0:
                
                     #   cX = int((M["m10"] / M["m00"]) * ratio)
                      #  cY = int((M["m01"] / M["m00"]) * ratio)
                        shape = sd.detect(c)
                        
                        if shape == "square":
                                
                                # multiply the contour (x, y)-coordinates by the resize ratio,
                                # then draw the contours and the name of the shape on the image
                                c = c.astype("float")
                               # c *= ratio
                                c = c.astype("int")
                                     
                                cv2.drawContours(gray, [c], -1, (0, 255, 0), 2)
                                #cv2.putText(gray, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                 #       0.5, (255, 255, 255), 2)
                
        

        cv2.imshow('video',gray)
        if cv2.waitKey(1)==27:# esc Key
                break

cap.release()    
cv2.destroyAllWindows()

