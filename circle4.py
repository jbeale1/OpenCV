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

    video_file = 'manual_2021-02-10_08.43.36_0.mp4' # new concentric targets
    #video_file = 'manual_2021-02-10_18.57.41_1.mp4' # more of the same
    #video_file = 'manual_2021-02-10_20.28.08_2.mp4' # yet more same
    video_file = 'manual_2021-02-11_00.18.44_0.mp4' # same, more
    
    #video_file = 'small_H1_out4.mp4'
    #video_file = 'output.mp4'

    showImage = False    # true to display detected frame
    showImage2 = False   # true to display mask image
    saveImage = False    # true to write each image to a file

    # --- configuration variables

    minGrey = 140   # greyscale threshold for "black" (0..255)
    maxGrey = 190
    #minGrey = 100   # greyscale threshold for "black" (0..255)
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
    
    # (x,y) variance data accumulator for 5 fiducials
    # Ad[3][0] holds data for fiducial #4, x coord.
    Ad = [ [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
           [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], 
           [[0, 0, 0], [0, 0, 0]] ]
    
    filename = argv[0] if len(argv) > 0 else video_file
    print("--- File: %s" % filename)    

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
    #print("count, dist")
    
    # expected target point area, Grey Thresh = 140
    
    #Tx,Ty,Td  = (609.967,110.887,107.2)  # UR1
    #Tx,Ty,Td  = (609.967,110.887,14.842) # UR4
    
    #Tx,Ty,Td = (609.406,277.353,111.3)   # CR1
    #Tx,Ty,Td = (612.792,278.917,062.8)   # CR2
    #Tx,Ty,Td = (612.792,278.917,045.8)   # CR3
    #Tx,Ty,Td = (612.792,278.917,015.8)   # CR3
    
    #Tx,Ty,Td = (609.406,277.353,15.1)    # CR4
    #Tx,Ty,Td = (608.944,447.198,110.2)   # LR1
    #Tx,Ty,Td = (608.944,447.198,15.5)    # LR4
    
    #Tx,Ty,Td = (409.783,270.707,128.4)   # BT1  beam tip, largest
    #Tx,Ty,Td = (409.572,272.067,102.2)   # BT2
    #Tx,Ty,Td = (409.783,270.707,87.4)   # BT3    
    #Tx,Ty,Td = (409.783,270.707,51.6)   # BT4 
    #Tx,Ty,Td = (409.783,270.707,38.9)    # BT5  beam tip, smallest
        
    #Tx,Ty,Td = (199.800,264.600,126.5)   # BL1 
    #Tx,Ty,Td = (199.800,264.600,100.2)   # BL2
    #Tx,Ty,Td = (199.800,264.600,86.0)    # BL3
    #Tx,Ty,Td = (199.800,264.600,50.2)    # BL4
    #Tx,Ty,Td = (199.370,264.399,38.8)    # BL5 
    
    Tx,Ty,Td = (1,1,38.8)    # nothing, nothing, nothing at all.
    
    # ==================== main loop ==============================
    while stop==False:     
     if (not pause):
      #cv.imwrite("frame1.png",frame)
      #exit() # DEBUG
      fc += 1      
      frame=frame[-600:,0:750,:] # mask off top of frame (date/time)
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       
             
      #gray = cv.medianBlur(gray, 3)  # Benefit accuracy? (no)
      #gray = unsharp_mask(gray)  # Benefit? (no)
      
      mask = cv.inRange(gray, minGrey, 255)  # hard threshold
      #mask = 255-cv.inRange(gray, minGrey, 255)
      #minGrey += 1  # scan threshold through range
      
      #ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
      #contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
      contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
      
      #print("Thresh: %d Contours: %d" % (minGrey,len(contours)))
      cframe = np.zeros(shape=frame.shape, dtype=np.uint8) # create blank BGR image      
      
      clist = []
      oF = []     # list of (x,y) center for outer ring of fiducials
      aC = []     # list of (x,y) centers for all contours
      aF = [[]]   # list of all centers of all fiducial rings
      
      
      for i in range(len(contours)): # loop over all contours
       M = cv.moments(contours[i])
       A = M['m00']  # area of contour
       if (A > 80) and (A < 40000):  # contour expected size?
         cnt = contours[i]
         perimeter = cv.arcLength(cnt,True)
         x,y,w,h = cv.boundingRect(cnt)
                   
         Db = max(w,h) # diameter of bounding circle
         D = math.sqrt(A*4/pi)     
         R = D/Db  # ratio of contour area to bounding circle area          
         
         cx = (M['m10']/A) # center of mass (cx,cy) of contour
         cy = (M['m01']/A)
         aC.append((cx,cy)) # save in list of all contour centers
           
         if (A > 8659) and (A < 40000):  # contour expected size?          
          
          if (R > 0.85): # typ > 0.92
            cv.drawContours(cframe, contours, i, (255,100,100), 1)
            #clist.append(cnt)  # save this contour
            oF.append((cx,cy))            
            #print("(%5.3f,%5.3f)" % (cx,cy))
            
            dx = abs(cx - Tx)  # 608.980,447.259
            dy = abs(cy - Ty)
            if (dx < 20) and (dy < 20):
            #if True:
              #print("%05.3f,%05.3f A=%5.1f D=%5.3f Ratio: %5.3f %5.3e,%5.3e" % (cx,cy,A,D,R,dx,dy),end="")
              if (abs(D-Td) < (0.1*D)):
                xSum = update(xSum,cx)
                ySum = update(ySum,cy)
                dSum = update(dSum,D)
                #print(" D=%5.3f" % D,end="")
              #print("")
              
              kp = cv.KeyPoint(cx,cy,10) # to draw center-indicator mark
              cframe = cv.drawKeypoints(
                cframe, [kp], np.array([]), (128,255,128), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)                                
                      
      #==  end of for i in contours[]  =============================
      
      # --------------------------------------------------------------------
      # Here we have the outer contours of the five fiducials in (clx[],cly[])
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
              
            
      #print(" ")                          
      cv.imshow("contours", cframe) # DEBUG - show contours
      
      #gray = cv.medianBlur(gray, 3)  # did not benefit accuracy      
      #gs = unsharp_mask(gray)  # sharpen image edges
 
# -------------------------------------------------
      #exit()
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
      
           
     #if (minGrey > maxGrey):
     #    stop=True
     key = cv.waitKey(1)
     if key == ord('q'):
           stop=True      
     if key == ord(' '):
         pause = not pause

# ------------------------------------------
# after all is calculated, show final averages

    for j in range(len(oF)):  # j is contour index number [0..4]
      (xMean,xStd,_) = finalize(Ad[j][0])
      (yMean,yStd,_) = finalize(Ad[j][1])
      print("F%d: (%5.3f,%5.3f) std: %5.3e,%5.3e" % 
        (j+1,xMean,yMean,xStd,yStd))
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

# =====================================================================
#   combine jpg images into mp4 video:
# ffmpeg -start_number 0 -i tl_00003_%05d.jpg -c:v libx264 -vf "fps=24,format=yuv420p" H1_out5.mp4
#   rescale video to new (x,y) dimensions:
# ffmpeg -i H1_out4.mp4 -vf scale=640:360,setsar=1:1 small_H1_out4.mp4
# 50.07 frames/cycle, 24 frames/s => T = 2.086 sec
