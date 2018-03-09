import numpy as np
import cv2
import sys



cap = cv2.VideoCapture(0)

while(True):
    _, img = cap.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #img = cv2.imread("zielfeld_seitlich.png")
    #img = cv2.imread("zielfeld_mit_Zeugs_2.jpg")
    #imS = cv2.resize(img, (480, 480))

   # img[img] = [255]
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ret,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    upper = 120
    lower_red = np.array([1,1,1])
    upper_red = np.array([upper,upper,upper])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    maskedImg = cv2.bitwise_and(img,img, mask= mask)

    thresh1_d = cv2.dilate(maskedImg,None)
    thresh1_e = cv2.erode(thresh1_d, None)

    
   # cv2.imshow('video_1',gray)
 
    dst = cv2.cornerHarris(np.float32(gray),2,3,0.04)
 
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
  
    edge_indices = np.transpose(np.where(dst>=0.01*dst.max()))

    #for i in edge_indices:
        #img[i[0],i[1]]=[0,255,0]

        
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    for i in np.int0(centroids):
        img[i[1],i[0]]=[0,255,0]


    res = res[1:]
    for i1 in range(len(res)/2):
        h_y = (res[i1][1] + res[len(res)-i1-1][1]) / 2
                  
    a = res[res[:,0].argsort()]
    h_x = (a[0][1] + a[len(a)-1][1]) / 2
    
    img[h_y][h_x-10:h_x+10]=[0,0,255]

    
    #cv2.line(img,(res[0][0], res[0][1]),(res[len(res)-1][0],res[len(res)-1][1]),(255,0,0),5)
   
    
   # cv2.line(img,(a[0][0], a[0][1]),(a[len(a)-1][0],a[len(a)-1][1]),(0,0,255),5)
    

    cv2.imshow('video',img)
    if cv2.waitKey(1)==27:# esc Key
        break

cap.release()    
cv2.destroyAllWindows()

