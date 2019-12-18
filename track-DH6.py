#!/usr/bin/env python3

'''
Monitor IP Camera for motion [<video_source>]
'''

# Python 3.6.6 and OpenCV 3.4.1
# by J.Beale 30-Oct-2018 - 18-Dec-2019

# for summary/montage thumbnail image:
# montage th_DH6*.png -tile 8x -geometry "1x1<+2+2" summary.png

from __future__ import print_function

import numpy as np
import cv2 as cv
import imutils  # http://www.pyimagesearch.com/2015/02/02/
import time
from datetime import datetime
from pathlib import Path  # (to check if log file exists)
import os       # to flush buffer to disk
import math     # math.floor()

# ==============================================================

CNAME = 'DH6'
UTYPE = "rtsp://"

# === shh: secret passwords are here ====
UPASS = "user:password"
FPASS = "ftp_username:ftp_password"
ftpDir = "ftp://my.ftpsite.com/cam3/"  # remote FTP directory to store images in
# === no secrets beyond this point =======


IPADD = "192.168.1.26"
PORT = "554"
URL2 = "/cam/realmonitor?channel=1&subtype=0"  # Dahua IP Cameras

# parameters tuned assuming stream at 15 fps
FDRATE=1          # frame decimation rate: only display 1 in this many frames
XFWIDE = 176       # width of resized frame
vThreshold = 1100   # how many non-zero velocity pixels for event
saveCount = 0      # how many images saved so far
vThresh= 15        # minimum "significant" velocity (was 30)
runLimit=2         # how many frames of no motion ends a run
validRunLength= 7  # minimum frames of motion for valid event
xcent = XFWIDE*0.38    # X ideal position for image
xcent2 = XFWIDE*0.50    # X ideal #2 position for image
xcent3 = XFWIDE*0.30    # X ideal #3 position for image
ycent = (XFWIDE*0.31)   # Y ideal position for image (was 0.19)
imageSaves = 1          # how many frames to save per detected event

# xcent = XFWIDE*0.4, ycent = XFWIDE*0.19
# => scaled w,h = (176, 99)
# Image w,h = 1920 1080  Scale=10.909091 (xc,yc) = 70.400000,33.440000


yTripline=255       # Y motion center g.t. this => driveway visitor
inDriveway = False # whether any motion this event was inside driveway
show_hsv = False
show_vt = True    # display velocity threshold map
logFname = 'Log_' + CNAME + '.txt'
fPrefix = CNAME + '_'     # prefix for saved image filename
fPrefix1 = CNAME + 'D_'   # used when event included driveway 
fPrefix2 = CNAME + 'b_'     # prefix for 2nd saved image filename
fPrefix3 = CNAME + 'c_'     # prefix for 3rd saved image filename
fPrefix12 = CNAME + 'bD_'   # used when event included driveway 
tPrefix = 'thumbs/th_' + CNAME
thumbWidth= 120   # max pixels wide for thumb
thumbAspect = 1.333;  # maximum aspect ratio for thumbnail
moT = 1.1         # Calc. dx/dt motion threshold for saving event
vidname= UTYPE + UPASS + "@" + IPADD + ":" + PORT + URL2
# ================================================================

def draw_flow(img, flow, step=10):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

 
def calc_v(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    #vs = (fx*fx+fy*fy)
    vs = (fx*fx)    
    # gf = np.zeros((h, w), np.uint8)
    gray = np.minimum(vs*64, 255) # gray.shape = (99, 176), float32
    retval, grf = cv.threshold(gray, vThresh, 255, 0)
    gr = grf.astype(np.uint8)
    gr = cv.dilate(gr, None, iterations=2)
    gr = cv.erode(gr, None, iterations=4)  # was 8
    gr = cv.dilate(gr, None, iterations=7) # was 5
    return gr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    #hsv[...,2] = np.minimum(v*4, 255)
    hsv[...,2] = np.minimum(v*64, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv.VideoCapture(vidname)  # open a video file or stream

    newlog = True
    logFile = Path(logFname)
    if logFile.is_file():
      newlog = False
    logf = open(logFname, 'a')
    tnow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("date,deltaFC,saveCount,mCount,fxAvg,mRun,Area,cx,cy,r")

    print("%s Start: %s  Newlog:%d" % (CNAME,tnow,newlog))
    if ( newlog ):
      hdr = "date,deltaFC,saveCount,mCount,fxAvg,mRun,Area,cx,cy,r,mFrac\n" # CSV column header
      logf.write(hdr)

    s = "# %s Start: %s\n" % (CNAME,tnow)
    logf.write(s)

    # cam = video.create_capture(fn)
    ret, prevRaw = cam.read()
    prev = imutils.resize(prevRaw, width=XFWIDE)
    bestImg = prevRaw
    tBest = time.localtime()

    w = int(cam.get(3))  # input image width
    h = int(cam.get(4))  # input image height
    fs = (1.0 * w) / XFWIDE # frame scaling factor
    print ("Image w,h = %d %d  Scale=%f (xc,yc) = %f,%f" % (w, h, fs, xcent, ycent))


    motionNow = False  # if we have current motion
    mRun = 0           # how many motion frames in a row
    frameCount = 1     # total number of frames read
    lastFC = 0         # previous motion frame count
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    cur_glitch = prev.copy()
    bsize=30           # buffer size, max elems in one run
    xpos=[]            # list of x positions empty
    xdelt=[]           # list of delta-x values
    s=[]               # list of data record strings starts empty
    rdist=[]           # radial distance to center frame
    minR = 1000        # force it high
    minR2 = 1000        # force it high
    minR3 = 1000        # force it high
    maxY = 0           # maximum Y motion center of an event
    mCountMax = 0      # largest mCount observed during this event

    # maxFrames = 40     # maximum frames in one motion record
    # frameElements = 10 # count data elements recorded per frame
    # mfrec = np.zeros((maxFrames,frameElements),dtype=float)  # array of motion frame data elements

    while True:
        #ret, img = cam.read()
        ret, imgRaw = cam.read()
        img = imutils.resize(imgRaw, width=XFWIDE)
        #w,h = cv.GetSize(img)  # width and height of image
        #xcent = w/2
        #ycent = h/2

        frameCount += 1
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)
        prevgray = gray
        vt = calc_v(flow)  # returns threshold map (mask) but uint8
        #print (vt.dtype)
        #print (vt.shape)
        vt0 = vt.copy()
        mCount = cv.countNonZero(vt)
        fx = flow[:,:,0]
        deltaFC = frameCount - lastFC # 1 or 2 during an event
        if (mCount > vThreshold): # significant motion detected this frame
          if (mCount > mCountMax):  # remember largest mCount during this event
            mCountMax = mCount
          motionNow = True
          im2,contours, hierarchy = cv.findContours(vt0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          #im2, contours, hierarchy = cv.findContours(vt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]
          # cv.drawContours(img, contours, -1, (0,255,0), 2)  # draw contours on image
          cnt = contours[0]  # select the largest contour
          M = cv.moments(cnt)
          Area = M['m00']       # area of contour
          cx = int(M['m10']/Area)
          cy = int(M['m01']/Area)
          dcx = (cx - xcent)
          dcx2 = (cx - xcent2)
          dcx3 = (cx - xcent3)
          dcy = (cy - ycent)
          r = np.sqrt(dcx*dcx + dcy*dcy)  # distance to center frame
          r2 = np.sqrt(dcx2*dcx2 + dcy*dcy)  # distance to center frame
          r3 = np.sqrt(dcx3*dcx3 + dcy*dcy)  # distance to center frame
          evR = cv.boundingRect(cnt)  # rect. around motion event
          y2=int(evR[1]+evR[3])   # y2 = y1 + height
          ttnow = datetime.now()
          # tnow = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # time to msec
          # datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
          tnow = ttnow.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # time to msec
          if (y2 > maxY):
              maxY = y2     # save largest Y value in this run
          if (r < minR):
              minR = r
              bestImg = imgRaw
              tBest = ttnow
              tsBest = tnow
              br = evR # rectangle around main contour
          if (r2 < minR2):
              minR2 = r2
              bestImg2 = imgRaw
              tBest2 = ttnow
          if (r3 < minR3):
              minR3 = r3
              bestImg3 = imgRaw
              tBest3 = ttnow

          #cx=0;
          fxSum,a,b,c = cv.sumElems(fx) # fx = flow along x direction
          fxAvg = fxSum / mCount
          xpos.append(cx)
          rdist.append(r)
          if (mRun > 0):
            xdelt.append(xpos[mRun]-xpos[mRun-1]) # remember each delta
          if (mRun >= validRunLength):
            v1 = np.mean(xdelt[0:3])
            v2 = np.mean(xdelt[mRun-4:mRun-1])
            if np.sign(v1) != np.sign(v2):
              deltaFC = runLimit+1 # force end of this run
              mCount = vThreshold - 1

          # Save data for this frame to log file
          # s.append("%s ,%d,%02d,%04d, %03d, %02d, %04d, %03d,%02d, %02d\n" % 
          s.append("%s ,%d,%02d,%04d, %03d, %02d, %04d, %03d,%02d, %02d" % 
            (tnow,deltaFC,saveCount,mCount,int(fxAvg*10),
             mRun,int(Area),cx,cy, int(r)) )
          # print(s[mRun],end='')  # without extra newline
          print(s[mRun],end='\n')  # without extra newline

          if (cy > yTripline):  # detected motion inside the driveway
            inDriveway = True   # flag this event for handling when its over
            # buf = "# Driveway" 
            # logf.write(buf)
            # print("# Driveway");  # action taken with driveway motion

          # logf.write(s)  # save in logfile
          lastFC = frameCount  # most recent frame with motion
          mRun += 1
          
        if (mCount <= vThreshold):
          if (deltaFC > runLimit):  # not currently in a motion event
            if (motionNow):  # was it active until just now??
              if mRun > 6:
                dxpos=[]
                for x in range(1, mRun):
                  dxpos.append(xpos[x]-xpos[x-1])
                xstd = np.std(dxpos)
                xavg = np.mean(dxpos)
                # dm1: average motion divided into 2 parts
                dm1 = (np.mean(xpos[mRun//2:mRun])
                   - np.mean(xpos[0:mRun//2])) / (0.5*mRun)
                mid2 = mRun//3  # find 1/3 and 2/3 points
                mid3 = mid2*2

                mn1 = np.mean(xpos[0:mid2])  # divided in 3 parts
                mn2 = np.mean(xpos[mid2:mid3])
                mn3 = np.mean(xpos[mid3:mRun])
                dm2 = (mn2-mn1)*3.0/mRun
                dm3 = (mn3-mn2)*3.0/mRun
                s1 = np.sign(dm1)  # real motion should be all same sign
                s2 = np.sign(dm2)
                s3 = np.sign(dm3)
                if (s1==s2 and s1==s3 and abs(dm1) > moT):
                  # write out all frame data records for this motion even to log file
                  for x in s:
                    fnum = int(x.split(',')[5]) # current frame number (from 0)
                    frac = float(fnum) / mRun   # fractional position based on total frame count
                    nstring = ("%s, %5.3f\n" % (x,frac))
                    logf.write(nstring)                  

                  # dts = time.strftime("%Y-%m-%d %H:%M:%S", tBest) # time & date string
                  # dts = time.strftime("%Y-%m-%d %H:%M:%S.%f", tBest)[:-3] # date+time to msec
                  buf = "# " + tsBest 
                  logf.write(buf)
                  print(buf,end='')
                  buf = (" , %d, %.2f,%.2f,%.2f, %3.2f, %03d, %d\n\n" % 
                     (mRun, dm2, dm1, dm3, xstd, maxY, mCountMax))
                  logf.write(buf)
                  print(buf,end='')  # without extra newline
                  #print(buf)
                  logf.flush()  # after event, actually write the buffered output to disk
                  os.fsync(logf.fileno())      
                  dt = tBest.strftime('%y%m%d_%H%M%S_%f')[:-3]
                  dt2 = tBest2.strftime('%y%m%d_%H%M%S_%f')[:-3]
                  dt3 = tBest3.strftime('%y%m%d_%H%M%S_%f')[:-3]
                  if (inDriveway):
                    fname3 = fPrefix1 + dt + ".jpg"  # full-size image, in driveway
                    fname32 = fPrefix12 + dt2 + ".jpg"  # full-size image
                  else:
                    fname3 = fPrefix + dt + ".jpg"  # full-size image
                    fname32 = fPrefix + dt2 + "b.jpg"  # full-size image
                    fname33 = fPrefix + dt3 + "c.jpg"  # full-size image

                  fname4 = tPrefix + dt + ".png"  # thumbnail image
                  # fname4 = tPrefix + dt + ".jpg"

                  x1=int(br[0]*fs)
                  y1=int(br[1]*fs)
                  x2=int((br[0]+br[2])*fs)
                  y2=int((br[1]+br[3])*fs)
                  # imutils.resize will not constrain both width and height
                  # so if input is too tall, need to reduce target width
                  dYin = (y2-y1)
                  dXin = (x2-x1)
                  Aspect = (dXin / dYin)
                  if (Aspect > thumbAspect):
                    thumbWidthEff = thumbWidth
                  else:
                    thumbWidthEff = math.floor(0.5 + (thumbWidth * (Aspect / thumbAspect)))

                  # print("%d,%d A=%5.3f width:%d" % (dXin, dYin, Aspect, thumbWidthEff))
                  thumbImg= imutils.resize(bestImg[y1:y2,x1:x2], 
                     width=thumbWidthEff)
                  cv.imwrite(fname4, thumbImg ) # active region
                  #cv.rectangle(bestImg,(x1,y1),(x2,y2),(0,255,0),1)  # draw rectangle on img
                  cv.imwrite(fname3, bestImg) # save best image
                  if (imageSaves > 1) and (tBest2 != tBest):
                    cv.imwrite(fname32, bestImg2) # save 2nd best image
                  if ((imageSaves > 2) and (tBest3 != tBest) and (tBest3 != tBest2)):
                    cv.imwrite(fname33, bestImg3) # save 3rd best image
                  # print("saved: " + fname3 + " , " + fname32 + " , " + fname33)
                  if (inDriveway):
                    os.spawnlp(os.P_NOWAIT, '/usr/bin/curl', 'curl', '-T', fname3, ftpDir + fname3, '--user', FPASS)


            inDriveway = False
            motionNow = False
            mRun = 0  # count of consecutive motion frames
            mCountMax = 0  # maximum mCount throughout event
            s=[]      # empty out list
            xpos=[]
            xdelt=[]
            rdist=[]
            minR = 1000        # force it high
            minR2 = 1000        # force it high
            minR3 = 1000        # force it high
            maxY = 0           # force it low
            
        if (motionNow):
          if (mRun > 4) and ((mRun-5) % 3) == 0:
            saveCount += 1
        
        if(frameCount % FDRATE) == 0:
          cv.imshow('Video_'+CNAME, img)  # raw image
          # cv.imshow('flow', draw_flow(gray, flow))
          if show_hsv:
            cv.imshow('HSV'+CNAME, draw_hsv(flow))
          if show_vt:
            cv.imshow('M_'+CNAME, vt)
          ch = cv.waitKey(1)
          
            
    logf.flush()  # after event, actually write the buffered output to disk
    #os.fsync()            
    logf.close()            
    cv.destroyAllWindows()
