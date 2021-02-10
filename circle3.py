#!/usr/bin/python3

# Detect black-on-white circles using contours; find center locations
# and intersection formed by lines connecting large and small dots

# works on Raspberry Pi with Python 3.7.3, OpenCV 4.5.1
# J.Beale 10-Feb-2021

import sys
import math
import time
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

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
# stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
    
# ====================================================================

def main(argv):
    
    default_file = '/home/pi/Pictures/circle4.jpg'
    #video_file = 'rp54_2021-02-08_20.27.33_1.mp4' # larger motion
    #video_file = 'rp54_2021-02-08_20.29.13_2.mp4' # smaller motion    
    #video_file = 'manual_2021-02-09_21.04.34_0.mp4'
    #video_file = 'manual_2021-02-09_21.18.37_2.mp4' # fairly small V
    #video_file = 'manual_2021-02-09_21.19.30_3.mp4' # very small V
    #video_file = 'H1_out6.mp4' # overnight 2/9
    #video_file = 'image_2021-02-10_08.25.35_0.jpg'
    video_file = 'manual_2021-02-10_08.43.36_0.mp4' # new concentric targets
    #video_file = 'small_H1_out4.mp4'
    #video_file = 'output.mp4'

    showImage = False    # true to display detected frame
    showImage2 = False   # true to display mask image
    saveImage = False    # true to write each image to a file

    # --- configuration variables

    minGrey = 100   # greyscale threshold for "black" (0..255)
    #minGrey = 100   # greyscale threshold for "black" (0..255)
    fc = 0    # video frame counter
    cT = 0    # total # circles detected    
    bc = 0     # how many frames in which multiple blobs detected
    fcount = 0 # how many frames saved to disk
    stop = False  # make true to end
    pause = False # if we are currently paused

# ---------------------------------------------------------------------
    pi = 3.14159265358979  # PI the constant


# Setup parameters for two detectors
    p1 = cv.SimpleBlobDetector_Params()  
    p2 = cv.SimpleBlobDetector_Params()

# Change thresholds
    p1.minThreshold = 60;  # 0-255 greyscale thresh (was 50)
    p2.minThreshold = 60;
    p1.maxThreshold = 150;  # 0-255 greyscale thresh (was 50)
    p2.maxThreshold = 150;
    p1.thresholdStep = 20;
    p2.thresholdStep = 20;
    p1.minRepeatability = 1;  # default 2
    p2.minRepeatability = 1; 

    # 1024x768: 74, 105
    # 1920x1080: 139.2, 195
    # size of circle features to find in image
    #dSmall = 74*1.875/3.0   # diameter in pixels of small circle
    #dBig = 105*1.875/3.0     # diameter in pixels of large circle
    
    # (47,-106)
    # 120: 49.879,48.727,107.254,106.876,104.308 120 x:-3.10 e-02 s:1.033e-07
    # 100: 40.582,41.327,107.604,107.023,104.510 100 x:-3.047e-02 s:8.614e-08
    #  80: 40.579,41.333,107.891,107.183,104.621 80  x:-3.008e-02 s:7.173e-08
    #  50: 46.853,40.545,108.453,107.825,105.091 50  x:-2.948e-02 s:5.267e-08
    #  50: 48.307,49.059,108.102,107.457,104.834,50  x:-2.247e-02 s:3.982e-07 (rfac = 0.9)
    #  20: 46.853,40.545,108.453,107.825,105.091 20  x: 5.529e-01 s:9.633e+00
    
    #  60: 50.697,49.760,108.661,107.918,105.574 60 x:-2.673e-02 s:1.822e-07 (w/sharp)
    #  60: 50.708,49.783,108.730,108.034,105.662,60 x:-2.735e-02 s:1.775e-07 (no sharp)


    dSmall = 45    # diameter in pixels of small circle
    dBig = 105     # diameter in pixels of large circle
    rfactor = 0.7  # factor minimum is less than mean (< 1.0) (0.7)
    
    aSmall = math.pow(dSmall/2,2) * pi
    aBig = math.pow(dBig/2,2) * pi
    p1.filterByArea = True
    p1.minArea = aSmall * rfactor
    p1.maxArea = aSmall / rfactor  # group 1: 2 small circles
    p2.filterByArea = True
    p2.minArea = aBig * rfactor
    p2.maxArea = aBig / rfactor  # group 2: 3 big circles
    
# Crowding limits    
    p1.minDistBetweenBlobs = dSmall/2;
    p2.minDistBetweenBlobs = dBig/2;    
    #p1.thresholdStep = 2; # default is 10
    #p2.thresholdStep = 2;


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
    print("Opening %s" % filename)

    video = cv.VideoCapture(filename)

    if not video.isOpened():
        print("Could not open file %s" % filename)
        sys.exit()

    # Attempt reading of just the first frame.
    ok, frame = video.read()
    if not ok:        
        print ('Error opening input file %s' % filename)
        sys.exit()

    #cv.imwrite("frame1.png",frame)  # write out Frame #1
    #exit() # DEBUG


    # csv file column headers
    print("count, dist")
    
    # ==================== main loop ==============================
    while stop==False:     
     if (not pause):
      #cv.imwrite("frame1.png",frame)
      #exit() # DEBUG
      fc += 1      
      frame=frame[-600:,:,:] # mask off top of frame (date/time)
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)              
      mask = cv.inRange(gray, minGrey, 255)
      #mask = 255-cv.inRange(gray, minGrey, 255)
      minGrey += 1  # scan threshold through range
      
      #ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
      #contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
      contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
      
      print("Thresh: %d Contours: %d" % (minGrey,len(contours)))
      cframe = np.zeros(shape=frame.shape, dtype=np.uint8) # create blank image      
      
      for i in range(len(contours)):
        M = cv.moments(contours[i])
        A = M['m00']  # area of contour
        if (A > 1100) and (A < 100000):  # contour expected size?
          cnt = contours[i]
          perimeter = cv.arcLength(cnt,True)
          x,y,w,h = cv.boundingRect(cnt)
          
          Db = max(w,h) # diameter of bounding circle
          D = math.sqrt(A*4/pi)     
          # CA = (Db*Db*pi/4) # area of circle with that diameter
          R = D/Db  # ratio of contour area to bounding circle area
          if (R > 0.90): # was 88
            cv.drawContours(cframe, contours, i, (255,100,100), 1)
            cx = (M['m10']/A) # center of mass (cx,cy)
            cy = (M['m01']/A)
            print("%05.3f,%05.3f  D=%5.3f Ratio: %5.3f" % (cx,cy,D,R))
            kp = cv.KeyPoint(cx,cy,10) # to draw center-indicator mark
            cframe = cv.drawKeypoints(
              cframe, [kp], np.array([]), (128,255,128), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)                        
            kp = cv.KeyPoint(cx,cy,16) # increase width of mark
            cframe = cv.drawKeypoints(
              cframe, [kp], np.array([]), (128,255,128), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)                        
        
        cv.imshow("Contours", cframe)
        #key = cv.waitKey(50)
        #time.sleep(0.5)
      
      #cv.drawContours(cframe, contours, -1, (255,150,150), 1) # all contours
      cv.imshow("contours", cframe) # DEBUG - show contours
      
      #gray = cv.medianBlur(gray, 3)  # did not benefit accuracy      
      #rows = gray.shape[0]  # not sure why this is here
         
      #gs = unsharp_mask(gray)  # sharpen image edges
      gs = gray
      
      # =========== do the blob detection =================
      kp1 = det1.detect(gs)  
      #gray = 255 - gray  # invert
      kp2 = det2.detect(255-gs)

      if showImage2:
        cv.imshow("mask", mask) # DEBUG - show mask image
        #cv.imshow("gray", gray) # DEBUG - show gray image

      if showImage:
        im1 = cv.drawKeypoints(
          frame, kp1, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im2 = cv.drawKeypoints(
          im1, kp2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
              
      cFound = len(kp1)
      for i in range(cFound):  # loop over blobs in set1.
        x = kp1[i].pt[0] # i is the index of the blob you want to get the position
        y = kp1[i].pt[1]
        print("%5.3f," % (kp1[i].size),end="") # size of blob in pixels
        kp1[i].size = 10 # rewrite detected blob to new diameter
        center=(int(x),int(y))         

      p = [0 for i in range(50)]
      cFound2 = len(kp2)  # work through group 2 (larger circles)
      for i in range(cFound2):  # loop over blobs in set2.          
        x = kp2[i].pt[0] # i is the index of the blob you want to get the position
        y = kp2[i].pt[1]
        p[i] = np.asarray((x,y))
        print("%5.3f," % (kp2[i].size),end="") # size of blob in pixels
        kp2[i].size = 10  # change apparent size of blob 

      if showImage:  # redraw new, smaller circles
        im2 = cv.drawKeypoints(
          im2, kp1, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im2 = cv.drawKeypoints(
          im2, kp2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
      # Three large circles with centers p[0..2]
      if (cFound > 1):        
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
        if (True):            
            if (cFound2 > 2): # if we have all 3 large circles
              
                # distance from point to line between two points
              d = np.linalg.norm(np.cross(p[1]-p[0], p[0]-p[2]))/np.linalg.norm(p[1]-p[0])
              dx = kp2[2].pt[0] - kp2[0].pt[0]
              dy = -(kp2[2].pt[1] - kp2[0].pt[1])
              angle2 = math.atan2(dy,dx) * 180/pi # angle from horizontal              
              #print("%5.3f, %5.3f"%(d,angle2),end=", ") 
              L1 = line([kp1[0].pt[0],kp1[0].pt[1]], 
                [kp1[1].pt[0],kp1[1].pt[1]]) # small circles
              L2 = line([kp2[0].pt[0],kp2[0].pt[1]], 
                [kp2[2].pt[0],kp2[2].pt[1]]) # large circles (2 on end)
              R = intersection(L1, L2)
              #cv.line(im2, (np.float32(kp1[0].pt[0]),np.float32(kp1[0].pt[1])), 
              #  (np.float32(R[0]),np.float32(R[1])), (0, 255, 0), 2,cv.LINE_AA)
              pt1 = cv.KeyPoint(R[0],R[1],14)      # make a keypoint      
              pt2 = cv.KeyPoint(R[0],R[1],16)      # make a keypoint      
              if showImage:
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
                #if (rDistC < 0):  # something wrong here
                #  pause = True

                dSum = update(dSum,rDistC)
                print ("%d,%5.4f" % (fc,rDistC)) # intersection distance 
                iDistStr = f'Position: {rDistC:06.4f} mm'
                if showImage:
                  cv.putText(im2, iDistStr, (100,650), cv.FONT_HERSHEY_SIMPLEX, 1, 
                     (200,200,200), 2, cv.LINE_AA)
                  jDistStr = f'scale: {umpp:5.2f} um/px'
                  cv.putText(im2, jDistStr, (100,720), cv.FONT_HERSHEY_SIMPLEX, 1, 
                     (50,200,50), 2, cv.LINE_AA)
              else:
                print ("  0  ",end=",")        
# -------------------------------------------------
      ok, frame = video.read()
      if not ok:
            break                

# --end pause loop -------------------------------------

     # Show keypoints
     if showImage:         
        cv.imshow("Blobs", im2)
     if saveImage and (bc>2):
        fout = f'det1{fcount:04d}.png'
        cv.imwrite(fout,im2)
        fcount += 1
        #exit()
      
           
     key = cv.waitKey(1)
     if key == ord('q'):
           stop=True      
     if key == ord(' '):
         pause = not pause

# ------------------------------------------
    (dmean, dvariance, dsampleVariance) = finalize(dSum)
    print("%d x:%6.3e s:%6.3e" % (p1.minThreshold, dmean, dvariance))
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

# =====================================================================
#   combine jpg images into mp4 video:
# ffmpeg -start_number 0 -i tl_00003_%05d.jpg -c:v libx264 -vf "fps=24,format=yuv420p" H1_out5.mp4
#   rescale video to new (x,y) dimensions:
# ffmpeg -i H1_out4.mp4 -vf scale=640:360,setsar=1:1 small_H1_out4.mp4
# 50.07 frames/cycle, 24 frames/s => T = 2.086 sec
