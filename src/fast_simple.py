import numpy as np
import cv2
import math

while(True):
    #img = cv2.imread("zielfeld_seitlich.png")
    img = cv2.imread("zielfeld_mit_Zeugs_2.jpg")
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(1)
    fast.setThreshold(100)

    kp = fast.detect(img,None)
    

    img3 = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
    cv2.imshow('video',img3)
        

    if cv2.waitKey(1)==27:# esc Key
        break

cv2.destroyAllWindows()
