import numpy as np
import cv2
import sys
import math


while(True):
    img = cv2.imread("zielfeld_seitlich.png")
    
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(gray,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(gray,kp,img)

    cv2.imshow('video',img2)
    if cv2.waitKey(1)==27:# esc Key
        break

cv2.destroyAllWindows()
