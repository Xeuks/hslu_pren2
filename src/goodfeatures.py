import numpy as np
import cv2



while(True):
    img = cv2.imread("zielfeld_seitlich.png")
  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,4,0.05,15)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
        
    cv2.imshow('fast_false.png',img)

    if cv2.waitKey(1)==27:# esc Key
        break

cv2.destroyAllWindows()
