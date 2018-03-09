import numpy as np
import cv2
import sys



cap = cv2.VideoCapture(0)

while(True):
    img = cap.read()[1]
    img = cv2.imread("zielfeld_seitlich.png")
    #img = cv2.imread("zielfeld_seitlich.png")
    #gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),5)


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


##    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,500, param1=60,param2=30,minRadius=20,maxRadius=0)

##    circles = np.uint16(np.around(circles))
##    for i in circles[0,:]:
##        # draw the outer circle
##        cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
##        # draw the center of the circle
##        cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

     # define range of blue color in HSV
    lower_blue = np.array([0,0,0])
    upper_blue = np.array([90,90,90])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img, lower_blue, upper_blue)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mask,50,150,apertureSize = 3)
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines is not None:
        for line in lines:
           # if np.array_equal(img[line[0][1], line[0][0]], [0,0,0]) or  np.array_equal(img[line[0][3], line[0][2]], [0,0,0]): 
             cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),1)


##    edged = cv2.Canny(gray, 200, 300)
##             
##                    # find the contours in the edged image and keep the largest one;
##                    # we'll assume that this is our piece of paper in the image
##    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
##    c = max(cnts, key = cv2.contourArea)
##             
##                    # compute the bounding box of the of the paper region and return it
##    marker =  cv2.minAreaRect(c)
##
##    box = np.int0(cv2.boxPoints(marker))
##
##    #min_x, min_y = numpy.min(box[0], axis=0)
##   #max_x, max_y = numpy.max(box[0], axis=0)
##    
##    cv2.drawContours(gray, [box], -1, (0, 255, 0), 2)

##    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##
##    gray = np.float32(gray)
##    dst = cv2.cornerHarris(gray,2,3,0.04)
##
##    #result is dilated for marking the corners, not important
##    dst = cv2.dilate(dst,None)
##
##    # Threshold for an optimal value, it may vary depending on the image.
##    img[dst>0.01*dst.max()]=[0,0,255]



    

    cv2.imshow('video',img)
    if cv2.waitKey(1)==27:# esc Key
        break

cap.release()    
cv2.destroyAllWindows()

