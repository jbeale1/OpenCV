#!/usr/bin/python3

# Detect black-on-white circles as blobs; find locations
# SimpleBlobDetector is 10x more accurate & consistent than HoughCircles
# in my tests using near-ideal input image quality
# even after optimizing all parameters to HoughCircles

# works with Pythone 3.7.3, OpenCV 4.5.1
# J.Beale 6-Feb-2021

import sys
import math
import cv2 as cv
import numpy as np

# ==================================

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

def main(argv):
    
    default_file = '/home/pi/Pictures/circle4.jpg'
    #video_file = '/home/pi/tracking/circle7.mp4'
    video_file = '/home/pi/tracking/circle9.mp4'

# ---------------------------------------------------------------------
# csv file column headers
    print("Frame, Dist, Davg, Dstd, angle")

# Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

# Change thresholds
    params.minThreshold = 128;
    #params.maxThreshold = 250;
    #params.thresholdStep = 30;

# Filter by Area.
    params.filterByArea = True
    params.minArea = 4000
    params.maxArea = 8000

# Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5
    params.maxCircularity = 1.0

# Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.maxConvexity = 1.0

# Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.9
    

# Create a detector with the parameters
    ver = (cv.__version__).split('.')
    #if int(ver[0]) < 3 :
    # 	detector = cv.SimpleBlobDetector(params)
    #else : 
    #detector = cv.SimpleBlobDetector_create()
    detector = cv.SimpleBlobDetector_create(params)
    


    sSum = (0, 0, 0)  # storage to calculate variance (radius)
    dSum = (0, 0, 0)  # storage to calculate variance (distance)
    bdSum = (0, 0, 0) # blob diameter 
    
    filename = argv[0] if len(argv) > 0 else video_file

    # Loads still image
    #src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Read video
    video = cv.VideoCapture(filename)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open file %s",filename)
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:        
        print ('Error opening file.')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        sys.exit()


#1248, 1, r=48.000, 1248, 2, r=47.000, d=146.055, 45, 49, 47.229, 0.702, 146.676, 2.635, 2 45-50-144 (4/0)
#1248, 1, r=48.000, 1248, 2, r=47.000, d=146.055, 45, 49, 47.227, 0.702, 146.646, 2.767, 2 45-50-152 (0/0)

#Failures = 5  Extras = 0

    pi = 3.14159265358979  # PI the constant
    minDist = 142  # was 296/2
    P1 = 100    # Hugh Param 1, was 100
    P2 = 30     # Hugh Param 2, was 30
    R1 = int(45)   # minimum valid radius
    R2 = int(50)  # maximum valid radius
    fc = 0    # video frame counter
    cT = 0    # total # circles detected
    rSum = 0  # sum of all circle radii
    rMin = 999 # max value radius found
    rMax = 0   # minimum value radius found
    fails = 0  # how many frames did not detect 2 circles
    extras = 0 # how  many frames found > 2 circles
    bc = 0     # how many frames in which multiple blobs detected
    stop = False  # make true to end
    pause = False # if we are currently paused
    
    while stop==False:
     if (not pause):
      # Read a new frame
      ok, frame = video.read()
      if not ok:
            break           
      fc += 1
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)              
      gray = cv.medianBlur(gray, 3)    
      #gray = cv.medianBlur(gray, 1)    
      rows = gray.shape[0]
      #print("rows = %d" % gray.shape[0]) # 1080
    
      # Sets pixels to white if in purple range, else will be set to black
      #mask = cv.inRange(gray, purpleMin, purpleMax)
      
      mask = cv.inRange(gray, 20, 255)
      invmask = 255 - mask
      #keypoints = detector.detect(invmask)
      kp = detector.detect(mask)
      #keypoints = detector.detect(gray)
      #cv.imshow("mask", mask)

      # Draw detected blobs as red circles.
      # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
      im_with_keypoints = cv.drawKeypoints(
        frame, kp, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
      cFound = len(kp)
      for i in range(cFound):  # loop over all blobs found. Hopefully, all circles
        x = kp[i].pt[0] #i is the index of the blob you want to get the position
        y = kp[i].pt[1]
        center=(int(x),int(y))
        cv.circle(im_with_keypoints, center, 10, (100, 0, 100), 3)
        
      if (cFound > 1):
        fc += 1
        x1=kp[0].pt[0]
        x2=kp[1].pt[0]
        y1 = kp[0].pt[1]
        y2 = kp[1].pt[1]
        if (x1 > x2):
            (x1,x2) = (x2,x1)
            (y1,y2) = (y2,y1)
        dx = x2 - x1
        dy = y1 - y2
        Bdist = math.sqrt(dx*dx + dy*dy)
        bdSum = update(bdSum,Bdist)
        angle = math.atan2(dy,dx) * 180/pi
        bc += 1
        if (bc > 2):            
            (bmean, bvariance, bsampleVariance) = finalize(bdSum)            
            print("%03d, %5.3f,%5.3f,%5.3f, %5.3f" % (fc,Bdist,bmean,bvariance,angle))


      # Show keypoints
      cv.imshow("Blobs", im_with_keypoints)
      
# =====================================================================      
    
      #cv.imshow("detected circles", frame)
      
     key = cv.waitKey(10)
     if key == ord('q'):
           stop=True      
     if key == ord(' '):
         pause = not pause
     #if (fc > 199):
     #      break           

    # print("Failures = %d  Extras = %d" % (fails,extras))
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
