#!/usr/bin/python3

# Detect black-on-white circles using contours; find center locations
# and intersection formed by lines connecting large and small dots

# works on Raspberry Pi with Python 3.7.3, OpenCV 4.5.1
# J.Beale 11-Feb-2021

import sys
import math
import time
import cv2 as cv
import numpy as np

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

def unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
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
    video_file = 'manual_2021-02-11_08.29.52_1.mp4' # moving parts

    showImage = False    # true to display detected frame
    showImage2 = False   # true to display mask image
    saveImage = False    # true to write each image to a file

    # --- configuration variables

    minGrey = 140  # greyscale threshold between black & white (0..255)
    
    fc = 0    # video frame counter
    cT = 0    # total # circles detected    
    bc = 0     # how many frames in which multiple blobs detected
    fcount = 0 # how many frames saved to disk
    stop = False  # make true to end
    pause = False # if we are currently paused

# ---------------------------------------------------------------------
    pi = 3.14159265358979  # PI the constant
    
    xSum = [0, 0, 0]  # storage to calculate variance (x)
    ySum = [0, 0, 0]  # storage to calculate variance (y)
    dSum = [0, 0, 0]  # contour diameter 
        
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
    
    """
    inframe = 0
    # tl_00004_00821.jpg
    infile = "tl_00004_%05d.jpg" % inframe
    print("--- File: %s" % infile)
    frame = cv.imread(infile)
    """        
    
    # ==================== main loop over image frames ==============
    while stop==False:     
     if (not pause):

      # (x,y) variance data accumulator for 5 fiducials
      # Ad[3][0] holds data for fiducial #4, x coord.
      Ad = [ [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
             [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
             [[0, 0, 0], [0, 0, 0]] ]
         
      #cv.imwrite("frame1.png",frame)
      #exit() # DEBUG
      fc += 1      # image frame counter
      
      #frame=frame[-650:,0:750,:] # mask off top of frame (date/time)
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       
             
      #gray = cv.medianBlur(gray, 3)  # Benefit accuracy? (no)
      #gray = unsharp_mask(gray)      # Benefit? (no)
      
      mask = cv.inRange(gray, minGrey, 255)  # hard threshold
      
      contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
      
      #print("Thresh: %d Contours: %d" % (minGrey,len(contours)))
      cframe = np.zeros(shape=frame.shape, dtype=np.uint8) # create blank BGR image      
            
      oF = []     # list of (x,y) center for outer ring of fiducials
      aC = []     # list of (x,y) centers for all contours
      aF = [[]]   # list of all centers of all fiducial rings      
      
      for i in range(len(contours)): # over all contours in frame
       M = cv.moments(contours[i])
       A = M['m00']  # area of contour  (3280/1024) = 3.203
       sf = 10.26  # 3.203 ^ 2
       if (A > 80*sf) and (A < 40000*sf):  # contour expected size?
         cnt = contours[i]
         perimeter = cv.arcLength(cnt,True)
         x,y,w,h = cv.boundingRect(cnt)
                   
         Db = max(w,h) # diameter of bounding circle
         D = math.sqrt(A*4/pi)     
         R = D/Db  # ratio of contour area to bounding circle area          
         
         cx = (M['m10']/A) # center of mass (cx,cy) of contour
         cy = (M['m01']/A)
         aC.append((cx,cy)) # save in list of all contour centers
           
            # contour of expected size and circularity (typ > 0.92)?
         if (A > 8659*sf) and (A < 40000*sf) and (R > 0.85): 
            cv.drawContours(cframe, contours, i, (255,100,100), 1)            
            oF.append((cx,cy))        # save this center point in list    
            
      #==  end of for i in contours[]  =============================
      
      # ---------------------------------------------------------------
      # Outer contours of the five fiducials in (clx[],cly[])
      for i in range(len(oF)):            
          cv.putText(cframe, str(i+1), (int(oF[i][0]),int(oF[i][1])), cv.FONT_HERSHEY_SIMPLEX, 1, 
              (255,255,255), 1, cv.LINE_AA)                        
      
      for i in range(len(aC)):  # scan through all contour center points
        for j in range(len(oF)):  # j is contour index number [0..4]
            dist = distance(aC[i],oF[j])
            if (dist < 4):
              # print("(%d,%d) %5.3f" % (i,j,dist))
              update(Ad[j][0],aC[i][0]) # record X coord
              update(Ad[j][1],aC[i][1]) # record Y coord
      # --------------------------------------------------------------
      fA = [] # list of final (x,y) centers of each fiducial
      for j in range(len(oF)):  # j is contour index number [0..4]
        (xMean,xStd,_) = finalize(Ad[j][0])
        (yMean,yStd,_) = finalize(Ad[j][1])
        fA.append((xMean,yMean))

      #for j in range(len(fA)):
      #  print("%5.3f,%5.3f, " % (fA[j][0],fA[j][1]),end="")
      #print("")  
      # Fiducial indexes: BT:0 BL:1 UR:4 CR:3 BR:2  fA[BT]=BoomTip
      BT=0; BL=1; UR=4; CR=3; BR=2
      
      # ----------------------------------------------------
            
      L1 = line([fA[BL][0],fA[BL][1]], 
        [fA[BT][0],fA[BT][1]]) # circles on boom
      L2 = line([fA[UR][0],fA[UR][1]], 
        [fA[BR][0],fA[BR][1]]) # fixed circles (2 on end)
      R = intersection(L1, L2)
      
      pt1 = cv.KeyPoint(R[0],R[1],14)      # make a keypoint      
      pt2 = cv.KeyPoint(R[0],R[1],16)      # make a keypoint      
      #if showImage:
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
        print ("%d,%5.4f,%5.4f,%5.4f" % (sec,rDistC,bDistC,tDistC))        
        
      #print(" ")                          
      cv.imshow("contours", cframe) # DEBUG - show contours
       
# -------------------------------------------------
      #exit()
      
      ok, frame = video.read()
      if not ok:
            break       
                  
      """                     
      inframe += 1          
      infile = "tl_00004_%05d.jpg" % inframe 
      try:
        #print(infile) 
        frame = cv.imread(infile)      
      except (FileNotFoundError, IOError):
        break
      if frame is None:  # we got nothing?
        break            # need to finish up then
      """      

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
# after all is calculated, show final averages

    """"
    print("Frames: %d" % fc)
    for j in range(len(oF)):  # j is contour index number [0..4]
      (xMean,xStd,_) = finalize(Ad[j][0])
      (yMean,yStd,_) = finalize(Ad[j][1])
      print("F%d: (%5.3f,%5.3f) std: %5.3e,%5.3e (%d)" % 
        (j+1,xMean,yMean,xStd,yStd,Ad[j][0][0]))
    """
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

# =====================================================================
#   combine jpg images into mp4 video:
# ffmpeg -start_number 0 -i tl_00005_%05d.jpg -c:v libx264 -vf "fps=24,format=yuv420p" H1_out8.mp4
#   rescale video to new (x,y) dimensions:
# ffmpeg -i H1_out4.mp4 -vf scale=640:360,setsar=1:1 small_H1_out4.mp4
# 50.07 frames/cycle, 24 frames/s => T = 2.086 sec
# Picam v1.3 FullRes: (3280x2464)
# find /dev/shm/t?.jpg | entr ./circle4.py /dev/shm/t7.jpg
