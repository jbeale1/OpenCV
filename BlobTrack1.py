# BlobTrack1.py
# background subtraction and blob tracking experiments
# OpenCV / Python    2015-06-28 J.Beale

import cv2           # OpenCV version 3.0.0
import numpy as np   # Numpy version 1.9
import sys

kernel5 = np.array([[0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0]]).astype(np.uint8)

kernel3 =np.array([[0, 1, 0],
                   [1, 1, 1],
                   [1, 1, 0]]).astype(np.uint8)
                   
maxpoints = 5                                # maximum number of blobs to track at once   
vfilt = 0.2                                  # stepwise velocity filter factor (low-pass filter)  
maxvel = 25                                  # maximum physically likely velocity (delta-X per frame)
xdistThresh = 50                             # how many pixels must a blob travel, before it becomes an event?
xc = np.float32(np.zeros(maxpoints))         # x center coordinates  
yc = np.float32(np.zeros(maxpoints))         # y center coordinates                     
xo = np.float32(np.zeros(maxpoints))         # x center, previous frame
xvel = np.float32(np.zeros(maxpoints))       # x velocity, instantaneous
xvelFilt = np.float32(np.zeros(maxpoints))   # x velocity (filtered by rolling average)  
xstart = np.float32(np.zeros(maxpoints))     # x starting point (for distance-travelled) 
xdist = np.float32(np.zeros(maxpoints))      # x distance-travelled since starting point
font = cv2.FONT_HERSHEY_SIMPLEX              # for drawing text
                   
cap = cv2.VideoCapture('2015-06-28-CarTest.mp4')  # compilation of test cases
#cap = cv2.VideoCapture(0)  # for example, a webcam

xproc = 640  # x,y resolution for processing
yproc = 480 # 640x360 for 16:9 aspect ratio source
record = True  # should we record video output?

voname = 'track-out2.avi' # name of video output to save

if (record):
  video = cv2.VideoWriter(voname, -1, 25, (xproc,yproc))  # open output video to record

history = 150
varThreshold = 18
detectShadows = True
fgbg = cv2.createBackgroundSubtractorMOG2(history,varThreshold,detectShadows)
# -----------------------------------------------------
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change blob detection thresholds
params.minThreshold = 200
params.maxThreshold = 255

params.minDistBetweenBlobs = 100

# Filter by Area.
params.filterByArea = True
params.minArea = 1200
params.maxArea = 40000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.02

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

print "bnum, x, y, rad, vel, dist"  # CSV file output header

while(1):
    ret, bframe = cap.read()

    if bframe is None:   
        if (record):
            video.release()    
        cap.release()
        cv2.destroyAllWindows()        
        sys.exit(1)

    frame = cv2.resize(bframe,(xproc,yproc))
    fgmask = fgbg.apply(frame)

    temp2 = cv2.erode(fgmask,kernel3,iterations = 2)    # remove isolated noise pixels with small kernel
    filtered = cv2.dilate(temp2,kernel5,iterations = 3) # dilate to join adjacent regions, with larger kernel
	
    inv = 255 - filtered  # invert black to white
    # Detect blobs.
    keypoints = detector.detect(inv)
    
    i = 0
    new = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR) # to allow us to draw colored circles on grey mask bitmap
    for k in keypoints:
        if (i < maxpoints):
          xc[i] = k.pt[0]   # x center of blob
          yc[i] = k.pt[1]   # y center of blob
          xs1 = int(k.pt[0]) # integer coords
          ys1 = int(k.pt[1])
          radius = int(k.size / 2)
          xvel[i] = xc[i] - xo[i]  # delta-x since previous frame
          if (abs(xvel[i]) > maxvel):  # force unreasonably large velocities (likely a glitch) to 0.0
            xvel[i] = 0
            xstart[i] = xc[i]  # reset x starting point to 'here'

          xdist[i] = xc[i] - xstart[i] # calculate distance this blob has travelled so far
          if abs(xvelFilt[i] - xvel[i]) < (2 + abs(xvelFilt[i]/2)):  # a sudden jump in value resets the average
            xvelFilt[i] = (vfilt * xvel[i]) + (1.0-vfilt)*xvelFilt[i]  # rolling average
          else:
            xvelFilt[i] = xvel[i]  # reset value without averaging
            
          print "%d, %5.3f, %5.3f, %5.1f,  %5.2f, %5.0f" % (i, xc[i], yc[i], k.size, xvelFilt[i], xdist[i])
          tstring = "%5.2f" % (xvelFilt[i])
          tstring2 = "%4.0f" % (xdist[i])
          if (abs(xdist[i]) > xdistThresh):
            cv2.putText(new,tstring,(xs1-30,ys1+80), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(new,tstring2,(xs1-30,ys1+95), font, 0.5,(255,255,255),2,cv2.LINE_AA)

          new = cv2.circle(new,(xs1,ys1),radius,[0,50,255],2, cv2.LINE_AA)  # round blob 
          xo[i] = xc[i]   # remember current x-center value for next frame
          i += 1 
    
    # Draw detected blobs as red circles.
    if (record):
        video.write(new)                    # save frame of output video
    cv2.imshow('source',frame)
    cv2.imshow("Keypoints", new)  # Show blob keypoints
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

if (record):
    video.release()    
cap.release()
cv2.destroyAllWindows()
