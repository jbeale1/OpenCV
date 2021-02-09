#!/usr/bin/python3

# Detect black-on-white circles as blobs; find center locations
# and intersection formed by lines connecting large and small dots

# SimpleBlobDetector is better than 10x more accurate & consistent
# than HoughCircles in my tests, using near-ideal input image quality
# even after separately optimizing all parameters to HoughCircles

# works on Raspberry Pi with Python 3.7.3, OpenCV 4.5.1
# J.Beale 8-Feb-2021

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

def intersection(L1, L2):  # find intersection point of 2 lines
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
        
def distance(P1, P2):  # signed distance between 2D points
    dx = P2[0] - P1[0]
    dy = P2[1] - P1[1]
    dist = np.sign(dy) * math.sqrt(dx*dx + dy*dy)
    return dist

# ====================================================================

def main(argv):
    
    default_file = '/home/pi/Pictures/circle4.jpg'
    #video_file = 'rp54_2021-02-08_20.27.33_1.mp4' # larger motion
    video_file = 'rp54_2021-02-08_20.29.13_2.mp4' # smaller motion    

    showImage = True     # true to display detected frame
    showImage2 = False   # true to display mask image
    saveImage = False    # true to write each image to a file

# ---------------------------------------------------------------------
    pi = 3.14159265358979  # PI the constant


# Setup parameters for two detectors
    p1 = cv.SimpleBlobDetector_Params()  
    p2 = cv.SimpleBlobDetector_Params()

# Change thresholds
    p1.minThreshold = 64;  # 0-255 greyscale threshold I presume
    p2.minThreshold = 64;

    # size of circle features to find in image
    dSmall = 74.2   # diameter in pixels of small circle
    dBig = 107     # diameter in pixels of large circle
    
    aSmall = math.pow(dSmall/2,2) * pi
    aBig = math.pow(dBig/2,2) * pi
    p1.filterByArea = True
    p1.minArea = aSmall * 0.7
    p1.maxArea = aSmall / 0.7  # group 1: 2 small circles
    p2.filterByArea = True    
    p2.minArea = aBig * 0.7
    p2.maxArea = aBig / 0.7  # group 2: 3 big circles

# Filter by Inertia
    p1.filterByInertia = True
    p1.minInertiaRatio = 0.7
    p2.filterByInertia = True
    p2.minInertiaRatio = 0.5
    
# Create a detector with the parameters
    #ver = (cv.__version__).split('.')
    det1 = cv.SimpleBlobDetector_create(p1)
    det2 = cv.SimpleBlobDetector_create(p2)
    
    sSum = (0, 0, 0)  # storage to calculate variance (radius)
    dSum = (0, 0, 0)  # storage to calculate variance (distance)
    bdSum = (0, 0, 0) # blob diameter 
    
    filename = argv[0] if len(argv) > 0 else video_file

    video = cv.VideoCapture(filename)

    if not video.isOpened():
        print("Could not open file %s" % filename)
        sys.exit()

    # Attempt reading of just the first frame.
    ok, frame = video.read()
    if not ok:        
        print ('Error opening input file %s' % filename)
        sys.exit()

    # --- configuration variables

    minGrey = 100   # greyscale threshold for "black" (0..255)
    fc = 0    # video frame counter
    cT = 0    # total # circles detected    
    bc = 0     # how many frames in which multiple blobs detected
    fcount = 0 # how many frames saved to disk
    stop = False  # make true to end
    pause = False # if we are currently paused

    # csv file column headers
    print("sA1, sA2, sZ1, sZ2, sZ3, Pdist, angle, Idist, Frame, Rdist, Davg, Dstd, DeltaAngle")
    
    # ==================== main loop ==============================
    while stop==False:
     if (not pause):

      #cv.imwrite("frame1.png",frame)
      #exit() # DEBUG

      fc += 1
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)              
      #gray = cv.medianBlur(gray, 3)  # did not benefit accuracy      
      rows = gray.shape[0]
      #print("rows = %d" % gray.shape[0])
         
      mask = cv.inRange(gray, minGrey, 255)
      #invmask = 255 - mask
      kp1 = det1.detect(mask)
      kp2 = det2.detect(mask)

      if showImage2:
        cv.imshow("mask", mask) # DEBUG - show mask image
        #cv.imwrite("mask1.png",mask)
        #exit() # DEBUG

      # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS shows size of blob
      if showImage:
        im1 = cv.drawKeypoints(
          frame, kp1, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im2 = cv.drawKeypoints(
          im1, kp2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
      #print("kp: %5.3f,%5.3f,%5.3f  " % (kp1[0].pt[0], kp1[0].pt[1], kp1[0].size))
      #pt1 = cv.KeyPoint(500.5+(fc/10.0),500,25)      # make a keypoint      
      #im2 = cv.drawKeypoints(
      #  im2, [pt1], np.array([]), (100,200,100), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
      cFound = len(kp1)
      for i in range(cFound):  # loop over blobs in set1.
        x = kp1[i].pt[0] # i is the index of the blob you want to get the position
        y = kp1[i].pt[1]
        if (bc > 2):
          print("%5.3f," % (kp1[i].size),end="") # size of blob in units of ?
        kp1[i].size = 10 # rewrite detected blob to new diameter
        center=(int(x),int(y))
        #if showImage:
          # cv.circle(im2, center, 3, (255, 0, 255), 2)
          #cv.putText(im2, str(i+1), center, cv.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), 1, cv.LINE_AA)
          

      p = [0 for i in range(50)]
      cFound2 = len(kp2)  # work through group 2 (larger circles)
      for i in range(cFound2):  # loop over blobs in set2.
        x = kp2[i].pt[0] # i is the index of the blob you want to get the position
        y = kp2[i].pt[1]
        p[i] = np.asarray((x,y))
        if (bc > 2):
          print("%5.3f," % (kp2[i].size),end="") # size of blob in units of ?
        kp2[i].size = 10  # change apparent size of blob 
        #center=(int(x),int(y))
        #if showImage:
        #  cv.circle(im2, center, 3, (0, 255, 0), 2)
        #  cv.putText(im2, (str(i+1)+str(i+1)), center, cv.FONT_HERSHEY_SIMPLEX, 1, 
        #    (255,255,0), 1, cv.LINE_AA)

      if showImage:  # redraw new, smaller circles
        im2 = cv.drawKeypoints(
          im2, kp1, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im2 = cv.drawKeypoints(
          im2, kp2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
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

        angle = math.atan2(dy,dx) * 180/pi
        bc += 1
        
        d02_mm = 10.660  # Distance from kp2[0] to kp2[2] is 10.660 mm
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
              #cv.line(im2, (np.float32(kp1[0].pt[0]),np.float32(kp1[0].pt[1])), 
              #  (np.float32(R[0]),np.float32(R[1])), (0, 255, 0), 2,cv.LINE_AA)
              pt1 = cv.KeyPoint(R[0],R[1],14)      # make a keypoint      
              pt2 = cv.KeyPoint(R[0],R[1],16)      # make a keypoint      
              im2 = cv.drawKeypoints(  # draw the intersection point
                im2, [pt1,pt2], np.array([]), (255,100,100), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
              
              if R:
                # Rdist in pixels, line-intersection to center lg.circle
                Rdist = distance(R,(kp2[1].pt[0],kp2[1].pt[1]))                
                # pDist: pixels between LargeCircle_0 and LargeCircle_2
                pDist = abs(distance((kp2[0].pt[0],kp2[0].pt[1]),(kp2[2].pt[0],kp2[2].pt[1])))
                mmpp = d02_mm / pDist # image scale in (mm per pixel)
                umpp = 1000*mmpp
                rDistC = Rdist * mmpp  # Rdist in units of mm
                bdSum = update(bdSum,rDistC)
                print ("%5.4f" % rDistC,end=",") # intersection distance 
                iDistStr = f'Position: {rDistC:06.4f} mm'
                cv.putText(im2, iDistStr, (100,750), cv.FONT_HERSHEY_SIMPLEX, 1, 
                   (200,200,200), 2, cv.LINE_AA)
                jDistStr = f'scale: {umpp:5.2f} um/pixel'
                cv.putText(im2, jDistStr, (560,750), cv.FONT_HERSHEY_SIMPLEX, 1, 
                   (0,0,0), 2, cv.LINE_AA)

              else:
                print ("0",end=",")

            if (bc > 4):
              (pmean, pvariance, psampleVariance) = finalize(bdSum)            
              print("%03d, %5.3f,%5.3f,%6.4f, %5.3f, %5.2f" % (fc,pDist,pmean,pvariance,angle2-angle,umpp))
        else:
            print()

     # Show keypoints
     if showImage:         
        cv.imshow("Blobs", im2)
     if saveImage and (bc>2):
        fout = f'det1{fcount:04d}.png'
        cv.imwrite(fout,im2)
        fcount += 1
        #exit()
      
# =====================================================================      
    
      #cv.imshow("detected circles", frame)
      
     key = cv.waitKey(1)
     if key == ord('q'):
           stop=True      
     if key == ord(' '):
         pause = not pause
         
     ok, frame = video.read()
     if not ok:
            break           

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

# =====================================================================
# combine det1xxxx.png images into mp4 video
# ffmpeg -start_number 0 -i det1%04d.png -c:v libx264 -vf "fps=24,format=yuv420p" H1_out.mp4
