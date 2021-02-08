#!/usr/bin/python3

# Detect black-on-white circles as blobs; find locations
# SimpleBlobDetector is 10x more accurate & consistent than HoughCircles
# in my tests using near-ideal input image quality
# even after optimizing all parameters to HoughCircles

# works with Pythone 3.7.3, OpenCV 4.5.1
# J.Beale 7-Feb-2021

import sys
import math
import cv2 as cv
import numpy as np

# ====================================================================

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

def line(p1, p2):  # construct a line from 2 points
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):  # find intersection of lines
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
        
def distance(P1, P2):  # distance between 2D points
    dx = P2[0] - P1[0]
    dy = P2[1] - P1[1]
    dist = np.sign(dy) * math.sqrt(dx*dx + dy*dy)
    return dist

def main(argv):
    
    default_file = '/home/pi/Pictures/circle4.jpg'
    #video_file = '/home/pi/tracking/circle9.mp4'
    #video_file = '/home/pi/tracking/dots3.mp4'
    video_file = '/home/pi/tracking/dots8a.mp4'

# ---------------------------------------------------------------------
    pi = 3.14159265358979  # PI the constant

# csv file column headers
    print("sA1, sA2, sZ1, sZ2, sZ3, Pdist, angle, Idist, Frame, Rdist, Davg, Dstd, DeltaAngle")

# Setup SimpleBlobDetector parameters.
    p1 = cv.SimpleBlobDetector_Params()
    p2 = cv.SimpleBlobDetector_Params()

# Change thresholds
    p1.minThreshold = 64;
    p2.minThreshold = 64;
    #params.maxThreshold = 250;
    #params.thresholdStep = 30;

# dots: size (diam) = 17.5, 26 pixels => area = pi/4 * d^2
# area = 240, 531

# dots: size = 37.5 => area = 1104
# dots: size = 20.26, 21.00 => area = 333
# Filter by Area.

# sz=70.777, sz=43.667, sz=45.129, sz=71.246, sz=70.426
    dSmall = 40*1.6   # diameter in pixels of small circle
    dBig = 71*1.6     # diameter in pixels of large circle
    aSmall = math.pow(dSmall/2,2) * pi
    aBig = math.pow(dBig/2,2) * pi
    p1.filterByArea = True
    p1.minArea = aSmall * 0.7
    p1.maxArea = aSmall / 0.7  # group 1: 2 small circles
    #p2.filterByArea = False
    p2.filterByArea = True
    p2.minArea = aBig * 0.7
    p2.maxArea = aBig / 0.7  # group 2: 3 big circles
    #p2.minArea = 794
    #p2.maxArea = 1620  # 70-110: smallest spots

# Filter by Inertia
    p1.filterByInertia = True
    p1.minInertiaRatio = 0.7
    p2.filterByInertia = True
    p2.minInertiaRatio = 0.5
    
# Create a detector with the parameters
    ver = (cv.__version__).split('.')
    det1 = cv.SimpleBlobDetector_create(p1)
    det2 = cv.SimpleBlobDetector_create(p2)
    
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
    showImage = True  # true to display detected frame
    showImage2 = True  # true to display mask image
    minGrey = 100   # greyscale threshold for "black" (0..255)
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
      #gray = cv.medianBlur(gray, 3)    
      #gray = cv.medianBlur(gray, 1)    
      rows = gray.shape[0]
      #print("rows = %d" % gray.shape[0]) # 1080
    
      # Sets pixels to white if in purple range, else will be set to black
      #mask = cv.inRange(gray, purpleMin, purpleMax)
      
      mask = cv.inRange(gray, minGrey, 255)
      #invmask = 255 - mask
      #keypoints = detector.detect(invmask)
      kp1 = det1.detect(mask)
      kp2 = det2.detect(mask)

      #keypoints = detector.detect(gray)
      if showImage2:
        cv.imshow("mask", mask) # DEBUG - show mask image

      # Draw detected blobs as red circles.
      # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
      im1 = cv.drawKeypoints(
        frame, kp1, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

      im2 = cv.drawKeypoints(
        im1, kp2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
      cFound = len(kp1)
      for i in range(cFound):  # loop over blobs in set1.
        x = kp1[i].pt[0] # i is the index of the blob you want to get the position
        y = kp1[i].pt[1]
        print("%5.3f," % (kp1[i].size),end="") # size of blob in units of ?
        center=(int(x),int(y))
        if showImage:
          cv.circle(im2, center, 3, (255, 0, 255), 2)
          cv.putText(im2, str(i+1), center, cv.FONT_HERSHEY_SIMPLEX, 1, 
            (255,0,255), 1, cv.LINE_AA)

      p = [0 for i in range(50)]
      cFound2 = len(kp2)  # work through group 2 (larger circles)
      for i in range(cFound2):  # loop over blobs in set2.
        x = kp2[i].pt[0] # i is the index of the blob you want to get the position
        y = kp2[i].pt[1]
        p[i] = np.asarray((x,y))
        print("%5.3f," % (kp2[i].size),end="") # size of blob in units of ?
        center=(int(x),int(y))
        if showImage:
          cv.circle(im2, center, 3, (0, 255, 0), 2)
          cv.putText(im2, (str(i+1)+str(i+1)), center, cv.FONT_HERSHEY_SIMPLEX, 1, 
            (255,255,0), 1, cv.LINE_AA)
        
      # Three large circles with centers p[0..2]
      if (cFound > 1):
        fc += 1
        x1=kp1[0].pt[0]
        x2=kp1[1].pt[0]
        y1 = kp1[0].pt[1]
        y2 = kp1[1].pt[1]
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
            if (cFound2 > 2): # if we have all 3 large circles
                # distance from point to line between two points
              d = np.linalg.norm(np.cross(p[1]-p[0], p[0]-p[2]))/np.linalg.norm(p[1]-p[0])
              dx = kp2[2].pt[0] - kp2[0].pt[0]
              dy = -(kp2[2].pt[1] - kp2[0].pt[1])
              angle2 = math.atan2(dy,dx) * 180/pi # angle from horizontal              
              print("%5.3f, %5.3f"%(d,angle2),end=", ") 
              L1 = line([kp1[0].pt[0],kp1[0].pt[1]], 
                [kp1[1].pt[0],kp1[1].pt[1]]) # small circles
              L2 = line([kp2[0].pt[0],kp2[0].pt[1]], 
                [kp2[2].pt[0],kp2[2].pt[1]]) # large circles (2 on end)
              R = intersection(L1, L2)
              
              if R:
                Rdist = distance(R,(kp2[1].pt[0],kp2[1].pt[1]))                
                print ("%5.3f" % Rdist,end=",") # intersection point
              else:
                print ("0",end=",")

            (bmean, bvariance, bsampleVariance) = finalize(bdSum)            
            print("%03d, %5.3f,%5.3f,%5.3f, %5.3f" % (fc,Bdist,bmean,bvariance,angle2-angle))
        else:
            print()

     # Show keypoints
     if showImage:
        cv.imshow("Blobs", im2)
        #cv.imwrite("blob1.png",im2)
        #exit()
      
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
