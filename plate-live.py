#!/usr/bin/env python3

"""
Scan IP Camera stream for license plates.
If plate seen, save image, position, time/date
Ignore repeated detection of a plate if it is not moving
J.Beale 7-JAN-2020
"""

import cv2
import sys
import json
import math
from datetime import datetime
from openalpr import Alpr
import regex as re # $ pip install regex

WINDOW_NAME  = 'openalpr'
CFILE="/home/john/CAM8/openalpr.conf"

fullInterval = 1.0   # seconds between full-frame saves
bRatioThresh = 1.3   # brightness step-change ratio threshold
mThresh = 5          # motion vector threshold (pixels)

fPrefix = "Pt_"
fPrefix2 = "D8_"
CNAME = 'DH8'  # camera name for output file
UTYPE = "rtsp://"
UPASS = "user:password"

IPADD = "192.168.1.28"
PORT = "554"
# URL2 = "/cam/realmonitor?channel=1&subtype=0"  # Dahua IP Camera, MAIN channel
URL2 = "/cam/realmonitor?channel=1&subtype=2"  # Dahua IP Cameras, subchannel B
vidname= UTYPE + UPASS + "@" + IPADD + ":" + PORT + URL2


def main():

  lastSaveTime = datetime.now()  # initialize last-image-save-time to prog. start time

  alpr = Alpr("us", CFILE, "/usr/share/openalpr/runtime_data/")
  if not alpr.is_loaded():
        print('Error loading OpenALPR')
        sys.exit(1)
  alpr.set_top_n(1)

  cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
  cv2.setWindowTitle(WINDOW_NAME, 'OpenALPR video test')
#  =========== now have a loaded instance of alpr; do stuff here

  cam = cv2.VideoCapture(vidname)  # open a video file or stream
  img_w = int(cam.get(3))  # input image width
  img_h = int(cam.get(4))  # input image height
  print ("Image w,h = %d %d" % (img_w, img_h))

  prlist = re.compile(r"\L<plates>", 
    plates=['7WXY20Z', '7WXYZ02', 'WXY202', '7WXY202', 'WXY20Z', 'WXYZ02', 
      'WXYZ0Z', '7WXY2Q2'])
  save = True  # always save the first image, for reference
  lastPstring = ""  # haven't seen any plates yet
  lastGarageDoor = True  # assume we're starting with door closed
  lastBright = 0.0       # average brightness of previous frame
  lastXcent = 0
  lastYcent = 0
  mvec=0

# ================== MAIN LOOP =====================================
  while ( True ):
    ret, img = cam.read()
    if not ret:
      print('VidepCapture.read() failed. Exiting...')
      sys.exit(1)

    ttnow = datetime.now()        # local real time when this frame was received
    avgbright = img.mean()        # average of all pixels (float)
    # print(avgbright)

    cv2.imshow(WINDOW_NAME, img) # show camera image

    results = alpr.recognize_ndarray(img)
    # print(".",end='')

    # print(results)
    jsonRes = (json.dumps(results, indent=2))
    jsonS = json.loads(jsonRes)

    rlist = results['results']
    pcount = len(rlist)             # how many plates we have found
    # print("Length = %d" % len(rlist) )
    if (pcount < 1):
      # print("No plates found, bright: %5.2f" % avgbright )
      pstring = ""   # null plate (nothing found)
    else:
      for i, plate in enumerate(rlist):
            cor1x = int(jsonS["results"][0]["coordinates"][0]["x"])
            cor1y = int(jsonS["results"][0]["coordinates"][0]["y"])
            cor2x = int(jsonS["results"][0]["coordinates"][2]["x"])
            cor2y = int(jsonS["results"][0]["coordinates"][2]["y"])
            xcent = (cor1x + cor2x) / 2
            ycent = (cor1y + cor2y) / 2
            dx = xcent - lastXcent
            dy = ycent - lastYcent
            mvec = math.sqrt(dx*dx + dy*dy)  # motion vector in pixels
            pcrop = img[cor1y:cor2y, cor1x:cor2x]  # crop of image containing plate
            cv2.imshow("plate", pcrop) # show just the plate

            p = plate['candidates'][0]
            pstring = p['plate'].upper()    # characters on license plate
            # if (pstring != "7WXY202") and (pstring != "WXY202"):


    if prlist.search(pstring):  # is this the usual garage-door view?
      garageDoor = True
    else:
      garageDoor = False

    # here: pstring holds the LP characters, or ""

    bRatio = (avgbright+0.01) / (lastBright+0.01)
    if (bRatio < 1.0):
      bRatio = 1/bRatio

    if (bRatio > bRatioThresh):
      bChange = True
    else:
      bChange = False

    if (mvec > mThresh):
      motion = True
    else:
      motion = False

    tnows = ttnow.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # time to msec
    if bChange :
      print("Brightness change: %5.3f , %s" % (avgbright,tnows))

    # save = not garageDoor   # save image if normal garage-door view not seen
    save = motion      # save image if motion of plate detected

    if (pstring==""):
      save = False   # but don't save image if no plate
      if (pstring != lastPstring):
        print("# Nothing: brightness %5.3f , %s" % (avgbright,tnows))  

    if save :
              print("%d,%d , %d,%d , " % (cor1x,img_h-cor1y,cor2x,img_h-cor2y),end='')
              print('%s , %5.2f , %5.2f , %5.2f , %s' % (pstring, p['confidence'], avgbright, mvec, tnows ))
              tnowf = ttnow.strftime("%Y-%m-%d_%H%M%S_%f")[:-3] # filename time to msec
              fname3 = fPrefix + tnowf + "_" + str(i) + ".jpg"  # full-size image
              # cv.imwrite(fname3, img) # save current image
              cv2.imwrite(fname3, pcrop) # save current image
              # print("saved: " + fname3)
              tSince = (ttnow - lastSaveTime).total_seconds()
              # print("seconds since last save = %5.3f" % tSince)
              if (tSince > fullInterval):
                fname4 = fPrefix2 + tnowf + "_" + str(i) + ".jpg"  # full-size image
                cv2.imwrite(fname4, img) # save full image
                lastSaveTime = ttnow

              save = False
              lastXcent = xcent
              lastYcent = ycent

    lastPstring = pstring  # remember what we saw before
    lastGarageDoor = garageDoor
    lastBright = avgbright      # remember last avg brightness

    c = cv2.waitKey(1) & 0xFF  # get input char if any
    if c == ord('q'):
      sys.exit(1)

# close up and quit
  alpr.unload()

# ======================================================

if __name__ == "__main__":
    main()
