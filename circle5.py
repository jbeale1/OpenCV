#!/usr/bin/python3

# Detect black-on-white circles using contours; find center locations
# and intersection formed by lines connecting large and small dots
# Currently runs on a single JPEG frame

# works on Raspberry Pi with Python 3.7.3, OpenCV 4.5.1
# J.Beale 11-Feb-2021

import sys
import math
import time
import cv2 as cv
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ====================================================================
# For a new value newValue, compute the new count, new mean, new M2.
#   mean accumulates the mean of the entire dataset
#   M2 aggregates the squared distance from the mean
#   count aggregates the number of samples seen so far

def update(D, newValue):
    count = D[0]
    mean = D[1]
    M2 = D[2]

    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    D[0] = count
    D[1] = mean
    D[2] = M2
    return (D)

# Retrieve the mean, variance and sample variance from collected data D
def finalize(D):
    count = D[0]
    mean = D[1]
    M2 = D[2] 
    if count < 2:
      return (float("nan"), float("nan"), float("nan"))
    else:        
      (mean, variance, sampleVariance) = (mean,M2/count,M2/(count - 1))
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

def distance(P1, P2):  # normal distance between 2D points
    dx = P2[0] - P1[0]
    dy = P2[1] - P1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    return dist
        
def distanceS(P1, P2):  # Y-direction signed distance between 2D points
    dx = P2[0] - P1[0]
    dy = P2[1] - P1[1]
    dist = np.sign(dy) * math.sqrt(dx*dx + dy*dy)
    return dist
    
# ====================================================================

def procImage(argv):
    
    output_file = '/dev/shm/Plog.csv'
    video_file = 'manual_2021-02-11_08.29.52_1.mp4' # moving parts

    showImage = False    # true to display detected frame
    showImage2 = False   # true to display mask image
    saveImage = False    # true to write each image to a file

    # --- configuration variables

    gStep = 5     # increment to change gray threshold between passes
    ctrGray = 150  # center threshold between black & white (0..255)
    
    sf = 10.26  # 1024->3280 image area scale factor: 3.203 ^ 2
    fc = 0    # video frame counter
    
    fcount = 0 # how many frames saved to disk
    stop = False  # make true to end
    pause = False # if we are currently paused

# ---------------------------------------------------------------------
    pi = 3.14159265358979  # PI the constant
    
    filename = argv[0] if len(argv) > 0 else video_file
    
    #print("n, mm, %s" % filename)    
    video = cv.VideoCapture(filename)
    if not video.isOpened():
        print("Could not open file %s" % filename)
        sys.exit()    
    ok, frame = video.read()  # read 1st frame
    if not ok:        
        print ('Error opening input file %s' % filename)
        sys.exit()
         
    
# ==================== main loop over image frames ==============

    while stop==False:     
     if (not pause):

      # (x,y) variance data accumulator for 5 fiducials
      # Ad[3][0] holds data for fiducial #4, x coord.
      Ad = [ [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
             [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
             [[0, 0, 0], [0, 0, 0]] ]
         
      fc += 1      # image frame counter
      
      # Picam 8MP FullRes: (3280x2464)
      # mask off top of frame (date/time) also empty area to right
      frame=frame[-2082:,0:2402,:] 
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       

      oF = []     # (x,y) center for outer ring, each of 5 fiducials
      aC = []     # (x,y) centers for all contours (may have garbage)
      
# ======== Do this for range of minGrey thresholds ==================
      
      for threshGray in range(ctrGray-4*gStep,ctrGray+4*gStep+1,gStep):
        mask = cv.inRange(gray, threshGray, 255)  # hard threshold
      
        contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        if (showImage):  # create blank BGR frame for contours
          cframe = np.zeros(shape=frame.shape, dtype=np.uint8) 
                  
        contourCount = len(contours)
        print("%d: %d" % (threshGray,contourCount)) # DEBUG
      
        for i in range(contourCount): # over all contours in frame
         M = cv.moments(contours[i])
         A = M['m00']  # area of contour  (3280/1024) = 3.203
         if (A > 80*sf) and (A < 40000*sf):  # contour expected size?
           cnt = contours[i]
           perimeter = cv.arcLength(cnt,True)
           x,y,w,h = cv.boundingRect(cnt)
                   
           Db = max(w,h) # diameter of bounding circle
           D = math.sqrt(A*4/pi)     
           R = D/Db  # ratio of contour area to bounding circle area          
         
           cx = (M['m10']/A) # center of mass (cx,cy) of contour
           cy = (M['m01']/A)
           aC.append((cx,cy)) # save in full list of all contour centers
         
           # Special case of outer contour on each target mark:  
           # contour of expected size and circularity (typ > 0.92)?
           if (threshGray == ctrGray):  # only at the ideal threshold
             if (A>8659*sf) and (A<40000*sf) and (R>0.85):            
               oF.append((cx,cy))   # store center of outer contour of each
               if (showImage): 
                 cv.drawContours(cframe, contours, i, (255,100,100), 1)            
      # end for threshGray...

# Above process for every set of contours, at every threshold              
# --------------------------------------------------------------------              
                        
      #==  end of for i in contours[]  =============================      
      # ---------------------------------------------------------------
      # Outer contours of just the five fiducials in oF[]
      if (showImage):
        for i in range(len(oF)):            
          cv.putText(cframe, str(i+1), (int(oF[i][0]),int(oF[i][1])), cv.FONT_HERSHEY_SIMPLEX, 1, 
              (255,255,255), 1, cv.LINE_AA)                        
      
      # scan through aC[] containing ALL contour centers (incl. junk)
      for i in range(len(aC)):  
        for j in range(len(oF)):  # j is contour index number [0..4]
            dist = distance(aC[i],oF[j])
            if (dist < 12):  # add this contour to existing stats file
              # print("(%d,%d) %5.3f" % (i,j,dist))
              update(Ad[j][0],aC[i][0]) # record X coord
              update(Ad[j][1],aC[i][1]) # record Y coord
              
      for j in range(len(oF)):  # DEBUG: show count of total averaged
        print("  Fid.%d: %d contours" % (j,Ad[j][0][0]))
        
      # --------------------------------------------------------------
      fA = [] # list of final (x,y) centers of each fiducial
      for j in range(len(oF)):  # j is contour index number [0..4]
        (xMean,xStd,_) = finalize(Ad[j][0])
        (yMean,yStd,_) = finalize(Ad[j][1])
        fA.append((xMean,yMean))  # record the final average values

      #for j in range(len(fA)):
      #  print("%5.3f,%5.3f, " % (fA[j][0],fA[j][1]),end="")
      #print("")  

      # Fiducial indexes: BT:0 BL:1 UR:4 CR:3 BR:2  fA[BT]=BoomTip
      BT=0; BL=1; UR=4; CR=3; BR=2  # specific fiducial indexes
            
      fiducialCount = len(fA)
      if (fiducialCount < 5):
          print("Failure: fiducial count %d" % fiducialCount)
          return(1)
      # ----------------------------------------------------
            
      L1 = line([fA[BL][0],fA[BL][1]], 
        [fA[BT][0],fA[BT][1]]) # circles on boom
      L2 = line([fA[UR][0],fA[UR][1]], 
        [fA[BR][0],fA[BR][1]]) # fixed circles (2 on end)
      R = intersection(L1, L2)
      
      pt1 = cv.KeyPoint(R[0],R[1],14)      # make a keypoint      
      pt2 = cv.KeyPoint(R[0],R[1],16)      # make a keypoint      
      if showImage:
        cframe = cv.drawKeypoints(  # draw the intersection point
          cframe, [pt1,pt2], np.array([]), (255,100,100), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)            
      
      # ----------------------------------------        
      d02_mm = 11.014  # Distance from UR to BR marks, per Inkscape SVG
      if R:
        # Rdist in pixels, line-intersection to center lg.circle
        Rdist = distanceS(R,(fA[CR][0],fA[CR][1]))                
        # pDist: pixels between UpperRight and BottomRight marks
        pDist = distance((fA[UR][0],fA[UR][1]),(fA[BR][0],fA[BR][1]) )
        mmpp = d02_mm / pDist # image scale in (mm per pixel)
        umpp = 1000*mmpp  # microns per pixel
        rDistC = Rdist * mmpp  # Rdist in units of mm
        
        # distance from center to bottom mark (should stay constant)
        bDistC = distance((fA[CR][0],fA[CR][1]),(fA[BR][0],fA[BR][1]))*mmpp
        # distance from center to top mark (should stay constant)
        tDistC = distance((fA[CR][0],fA[CR][1]),(fA[UR][0],fA[UR][1]))*mmpp
        
        sec = time.time() # real time, seconds since epoch

        # beam (mm) above center & 2 hopefully fixed distances
        #print ("%d,%5.4f,%5.4f,%5.4f" % (fc,rDistC,bDistC,tDistC))
        
        fout= open(output_file,"a+")      # append to log file
        fout.write("%d,%5.4f,%7.5f,%7.5f,%5.3f\n" % 
          (sec,rDistC,bDistC,tDistC,umpp))
        fout.close

# -------------------------------------------------
      
      ok, frame = video.read()
      if not ok:
            break       
                  
# --end pause loop -------------------------------------

     # Show keypoints
     if showImage:         
        cv.imshow("Detections", im2)
      
     #key = cv.waitKey(1)
     #if key == ord('q'):
     #      stop=True      
     #if key == ord(' '):
     #    pause = not pause

# ------------------------------------------
    
    return 0

# --------------------------------------------------------------
# Run procImage() program whenever specific file is changed
class Watcher:
    DIRECTORY_TO_WATCH = "/dev/shm"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'modified':
            if (event.src_path == '/dev/shm/tflag.txt'):              
              print("Got flag %s." % event.src_path)
              procImage(['/dev/shm/t7.jpg'])              
              
if __name__ == "__main__":
    w = Watcher()
    w.run()
    #procImage(sys.argv[1:])

# =====================================================================
#  NOTES :  combine jpg images into mp4 video:
# ffmpeg -start_number 0 -i tl_00005_%05d.jpg -c:v libx264 -vf "fps=24,format=yuv420p" H1_out8.mp4
#   rescale video to new (x,y) dimensions:
# ffmpeg -i H1_out4.mp4 -vf scale=640:360,setsar=1:1 small_H1_out4.mp4
# 50.07 frames/cycle, 24 frames/s => T = 2.086 sec
# Picam 8MP FullRes: (3280x2464)
# find /dev/shm/t?.jpg | entr ./circle4.py /dev/shm/t7.jpg >> Feb11_Log.csv &
"""
Gray: Contour Count
130: 161
135: 161
140: 142
145: 130
150: 134 <== midpoint
155: 164
160: 199
165: 195
170: 153
"""
