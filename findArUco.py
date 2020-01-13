#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import math   # pow(), sqrt()
from datetime import datetime  # time/date stamp

dThresh = 3  # distance in pixels away from "home" that is significant
#dThresh = -3  # distance in pixels away from "home" that is significant
xavg = [18, 70]  # starting (x,y) position of fiducial
yavg = [109, 11]

stream="rtsp://user:pass@192.168.1.28:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(stream)  # open a video file or stream
img_w = int(cap.get(3))  # input image width
img_h = int(cap.get(4))  # input image height
print ("Image w,h = %d %d" % (img_w, img_h))
ret, img = cap.read()  # read first frame
if not ret:
    print('VideoCapture.read() failed. Exiting...')
    sys.exit(1)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
# parameters.adaptiveThreshConstant = 10
parameters.minMarkerPerimeterRate = 0.9 # 0.35works ok
parameters.maxMarkerPerimeterRate = 2.0 # 0.625 works ok
# parameters.aprilTagQuadSigma = 0.8           # gaussian filter to blur noise
print("Min,Max: %f %f" % (parameters.minMarkerPerimeterRate, parameters.maxMarkerPerimeterRate) )

idx = 0   # frame index
countLast = 0      # no previous marks detected
eventLast = True   # force readout on first pass through loop

while(True):
    # Capture frame-by-frame
    ret, fullframe = cap.read()
    ttnow = datetime.now()        # local real time when this frame was received

    idx += 1
    #frame = fullframe[int(img_h*0.1):int(img_h*0.54), int(img_w*0.425):int(img_w*0.50)]
    #frame = fullframe[int(img_h*0.2):int(img_h*0.44), int(img_w*0.4):int(img_w*0.48)]
    frame = fullframe[int(img_h*0.2):int(img_h*0.428), int(img_w*0.405):int(img_w*0.467)]
    g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.blur(g1,(3,3))
    # g2 = g1
    # gray = cv2.equalizeHist(g2)  # do auto-levels
    gray = g2

    #print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print("%d %d" % (ids[0][0],ids[1][0]))
    detMarks = 0   # how many valid marks detected
    rdist = [0,0]  # distance from expected position
    xnew = [0,0]
    ynew = [0,0]
    if (ids is not None):
        for i in range(len(ids)):
            if (ids[i][0] == 17):
                detMarks += 1   # how many valid mark symbols
                # xnew[i] = corners[i][0][0][0]
                ynew[i] = corners[i][0][0][1]
        ynew.sort(reverse=True)  # need a consistent sort order
        for i in range(detMarks):
          dy = ynew[i]-yavg[i]
          rdist[i] = abs(dy)  # distance from expected spot
          frame2 = aruco.drawDetectedMarkers(frame, corners)
    else:
        pass  # print("#")

# =================================================

    if (detMarks < 2) or (rdist[0] > 2.5) or (rdist[1] > 2.5):
        event = True
    else:
        event = False

    if (event or eventLast):
        printLine = True
    else:
        printLine = False

    if (detMarks == 0) and (countLast == 0):
        printLine = False   # don't print more than one consecutive 0 line

    if printLine :
        tnows = ttnow.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # time to msec

        print("Found: %d  %s  " % (detMarks,tnows),end="")
        for i in range(detMarks):
          print("%05.2f,%05.2f : %3.1f ,  " % (xnew[i],ynew[i],rdist[i]), end="" )
        print("")
        if (detMarks == 1):
          tnowf = ttnow.strftime("%Y-%m-%d_%H%M%S_%f")[:-3] # filename time to msec
          fname3 = "FD_" + tnowf + ".jpg"  # plate crop image
          # cv2.imwrite(fname3, raw) # save current crop of plate

    eventLast = event      # whether motion event detected
    countLast = detMarks   # how many patterns detected

    #print(rejectedImgPoints)
    # Display the resulting frame
    if (ids is None):
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',frame2)
        # cv2.imshow('frame2',g2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
