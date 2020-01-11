# detect square features (outside of ArUco marks) of a specific size

import cv2
import numpy as np
import math

# from matplotlib import pyplot as plt
stream="rtsp://usr:pass@192.168.1.28:554/cam/realmonitor?channel=1&subtype=0"
# stream = "/home/john/Videos/2020-JAN-10-TRB-leave-daylight.m4v"

# raw = cv2.imread('D8_2020-01-10_084639_952_0.jpg')
#img = cv2.imread('noisy2.jpg',0)
#img = cv2.imread('noisy1.jpg',0)

cap = cv2.VideoCapture(stream)  # open a video file or stream
img_w = int(cap.get(3))  # input image width
img_h = int(cap.get(4))  # input image height
# print ("Image w,h = %d %d" % (img_w, img_h))
print("x , y") # CSV column headers
ret, img = cap.read()  # read first frame
if not ret:
    print('VideoCapture.read() failed. Exiting...')
    sys.exit(1)


#img_h = 720
#img_w = 1280

caMin = 4*2000   # minimum valid contour area
caMax = 4*3000  # max valid contour area
idx=0
box = []
# crop = img[int(img_h*0.1):int(img_h*0.54), int(img_w*0.425):int(img_w*0.50)]
#crop = img[int(img_h*0.3):int(img_h*0.54), int(img_w*0.425):int(img_w*0.50)]
while(True):
    # Capture frame-by-frame
    ret, fullframe = cap.read()
    idx += 1

    raw2 = fullframe[int(img_h*0.1):int(img_h*0.54), int(img_w*0.425):int(img_w*0.50)]
    raw = cv2.resize(raw2,None,fx=2, fy=2)

    grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(grey,5)
    img = cv2.medianBlur(grey,3)

#    blocksize = 99
#    Aconst = 4
    blocksize = 49
    Aconst = 4

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,blocksize,Aconst)
    # cv2.imshow('AdaptThr',th3)

    # image, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cgood = []
    idx = 0            # target box: area ~2582, AoP ~0.257? perimeter 200 ?
    rFound = False
    for c in contours:
        area = cv2.contourArea(c)
        if (area > caMin) and (area < caMax):
            perimeter = cv2.arcLength(c,True)  # calculate perimeter
            p2 = math.pow(perimeter/2,2)
            AoP = area / p2
            if (AoP > 0.19): #ideal square: 0.25
              rFound = True
              #cgood.append(c)
              # x,y,w,h = cv2.boundingRect(c)
              rect = cv2.minAreaRect(c)
              box = cv2.boxPoints(rect)
              # print(box[0])
              box = np.int0(box)
              cgood.append(box)
              # print("%d %d %4.3f" % (idx,area, AoP))
              idx += 1

    # print(" ")
    #img = cv2.drawContours(raw, contours, -1, (0,255,0), 1)
    # img = cv2.drawContours(raw, cgood, -1, (0,255,0), 1)
    # img = cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
    raw2 = raw
    if rFound:
      for box in cgood:
        raw2 = cv2.drawContours(raw2,[box],0,(0,0,255),2)
        M = cv2.moments(box)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])
        print("%05.2f , %05.2f" % (cx,cy))
        raw2 = cv2.circle(raw2,(int(cx),int(cy)), 4, (0,255,0), -1)
        #raw2 = cv2.drawContours(raw,[box],0,(0,0,255),2)

    cv2.imshow('Contours', raw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
