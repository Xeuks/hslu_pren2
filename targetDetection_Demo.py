import numpy as np
import cv2
import math

# Get webcam stream
cap = cv2.VideoCapture(0)

while(True):

    # Read image stream, returns multiple values
    # _, = discard first return value (not used)
    _, img = cap.read()

    # Convert image into black and white
    #img = cv2.imread("zielfeld_mit_Zeugs_2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # Disable nonmaxSuppression, set threshhold for corner detection
    fast.setNonmaxSuppression(1)
    fast.setThreshold(100)

    # Execute the FAST algorithm
    kp = fast.detect(img,None)

    # Create a new image with added keypoints and show it
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    cv2.imshow('video1',img2)
     
    # Variables needed for the following calculations
    prev_d_len = 0
    prev_x_len = 0
    prev_y_len = 0
    threshold = 10
    required_matches = 4
    matched = 0
    i = 0
    is_first = True

    # Calculating distances (keypoints are sorted by y-values)
    # Variable y is not being used
    while i < len(kp)-2:

        # Current keypoint
        cur = kp[i].pt
        # Diagonally nearest keypoint
        near_d = kp[i+2].pt
        # Nearest keypoint on x axis
        near_x = kp[i+1].pt
        # Nearest keypoint on y axis
        near_y = kp[len(kp)-i-1].pt

        # Calculate diagonal distance
        d_len = math.fabs(cur[0]-near_d[0])+math.fabs(cur[1]-near_d[1])
        # Calculate distance on x axis
        x_len = math.fabs(cur[0]-near_x[0])+math.fabs(cur[1]-near_x[1])
        # Calculate distance on y axis
        #y_len = math.fabs(cur[0]-near_y[0])+math.fabs(cur[1]-near_y[1])

        # Check if the current keypoint is the first one
        if is_first:
            cont_check = True
            is_first = False
        # Otherwise, check for continuity by comparing calculated values
        else:
            # True if previous length + threshhold is greater than current distance
            # and if previous length - threshhold is less than current distance (getting smaller)
            cont_check = (prev_d_len+threshold > d_len) and (prev_d_len-threshold < d_len)
            # Check if all x added up gives the total length on the x axis
            should_x_len = x_len + 2 * math.fabs(kp[i].pt[0]-kp[i-2].pt[0])
            # True if all conditions are met
            cont_check = cont_check and (prev_x_len+threshold > should_x_len) and (prev_x_len-threshold < should_x_len)
            #should_y_len = y_len + 2 * math.fabs(kp[i].pt[1]-kp[i-2].pt[1])
            #cont_check = cont_check and (prev_y_len+threshold > should_y_len) and (prev_y_len-threshold < should_y_len)
             
        if cont_check == True:
            prev_d_len = d_len
            prev_x_len = x_len
            prev_y_len = y_len
            matched += 1
            i = i+2
           
            if matched >= required_matches:
                print "ok"
                
                h_x = int(kp[i].pt[0]) + int(math.fabs((kp[i].pt[0]-kp[i+1].pt[0])/2))
                h_y = int(kp[i].pt[1]) + int(math.fabs((kp[i].pt[0]-kp[i+1].pt[0])/2))

                ref_len_p = math.fabs(kp[i].pt[0]-kp[i+1].pt[0])
                ref_len_cm = 2.6

                height, width = img.shape[:2]
                mid_x = int(width /2)

                act_len = (math.fabs(mid_x-h_x) / ref_len_p)*ref_len_cm
                #width_len = width/ref_len_p*ref_len_cm

                t_len = math.fabs(kp[i-2].pt[0]-kp[i-2+1].pt[0])
                length = t_len/ref_len_p*ref_len_cm

                print act_len, length

                
                img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
                #img3 = cv2.drawKeypoints(img3, [kp[i-2], kp[i-2+1]], None, color=(0,255,0))
                for idx in range(10):
                    img3[h_y-5+idx][h_x] =[0,0,255]

                img3[h_y][h_x-10:h_x+10] =[0,0,255]
               
                #img3[int(height/2)][mid_x-10:mid_x+10] =[0,0,255]
                   
                cv2.imshow('video',img3)
                break

        else:
          
            is_first = True
            matched = 0
            prev_d_len = 0
            prev_x_len = 0
            prev_y_len = 0
            i+=1
    
    if cv2.waitKey(1)==27:# esc Key
        break
    
cap.release() 
cv2.destroyAllWindows()
