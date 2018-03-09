import numpy as np
import cv2
import math

#cap = cv2.VideoCapture(0)
video = './bla/20180309_102034.mp4'
image =  "bla/20180308_092251.jpg"
cap = cv2.VideoCapture(video)

while(True):
    ret, img = cap.read()
   # img = cv2.imread(image)
    img = cv2.resize(img, (750, 600)) 

   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
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

    img[res[:,3],res[:,2]] = [0,255,0]
    
    prev_d_len = 0
    prve_x_len = 0
    prve_y_len = 0

    prev_h_y = 0
    prev_h_x = 0
    
    threshold = 10
    required_matches = 4
    matched = 0
    i = 0
    is_first = True

    while i < len(res)-2:
        cur = res[i]
        near_d = res[i+2]
        near_x = res[i+1]

        d_len = math.fabs(cur[3]-near_d[3])+math.fabs(cur[2]-near_d[2])
        x_len = math.fabs(cur[3]-near_x[3])+math.fabs(cur[2]-near_x[2])

        if is_first:
            cont_check = True
            is_first = False
        else:
            cont_check = (prev_d_len+threshold > d_len) and (prev_d_len-threshold < d_len)
            should_x_len = x_len + 2 * math.fabs(res[i][3]-res[i-2][3])
            cont_check = cont_check and (prve_x_len+threshold > should_x_len) and (prve_x_len-threshold < should_x_len)
             
        if cont_check == True:
            prev_d_len = d_len
            prve_x_len = x_len

            matched += 1
            i = i+2
           
            if matched >= required_matches:
                print("ok")
                p1x = res[i-1,3]
                p1y = res[i-1,2]
                p2x = res[i,3]
                p2y = res[i,2]
                
                prev_h_x = h_x = int(p1x) + int(math.fabs((p2x-p1x)/2))
                prev_h_y = h_y = int(p1y) + int(math.fabs((p2y-p1y)/2))

                
                ref_len_cm = 11
                ref_len_p = math.fabs(p2x-p1x)

##                height, width = img.shape[:2]
##                mid_x = int(width /2)
##
##                act_len = (math.fabs(mid_x-h_x) / ref_len_p)*ref_len_cm
##               # width_len = width/ref_len_p*ref_len_cm
##
##                t_len = math.fabs(kp[i-2].pt[0]-kp[i-2+1].pt[0])
##                length = t_len/ref_len_p*ref_len_cm
##
##                print(act_len, length)

                img[h_x,h_y]= [0,0,255]
                img[res[i-1,3],res[i-1,2]] = [0,0,255]
                img[res[i,3],res[i,2]] = [0,0,255]

                   
                cv2.imshow('video',img)
                #break

##                while cv2.waitKey(1)!=27:
##                    pass
                break

        else:
            img[prev_h_x,prev_h_y]= [0,0,255]
            cv2.imshow('video',img)
            
            is_first = True
            matched = 0
            prev_d_len = 0
            prve_x_len = 0
            prve_y_len = 0
            i+=1

    

    
    
    
    if cv2.waitKey(1)==27:# esc Key
        break

cap.release()    
cv2.destroyAllWindows()
