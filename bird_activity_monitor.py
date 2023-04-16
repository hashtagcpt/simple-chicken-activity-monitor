# import the necessary packages
import argparse
import datetime
import imutils
import datetime
import time
import numpy
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(1) # change to 0 for webcam
    time.sleep(0.25)
    print(camera)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None


counter = 0 # frame reset counter
AreaSum = 0 # motion area summary for each minute
RegionSum = 0 # number of regions summary for each minute

# pause flag
monitorPause = False
text = '' # empty text flag
todayIs = datetime.datetime.now().strftime("%Y-%m-%d")
dataFile = open(todayIs + ".csv",'w')
initTime = datetime.datetime.now()
# loop over the frames of the video
while True:
    # if it is a new day make a new data file
    if todayIs != datetime.datetime.now().strftime("%Y-%m-%d"):
        dataFile.close()
        todayIs = datetime.datetime.now().strftime("%Y-%m-%d")
        dataFile = open(todayIs + ".csv",'w')
        dataFile.write('Time, Regions, Area\n')
                
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    regions = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area.append(w*h)

        # get a summary measure of the area and number of contours
        n_area_sum = numpy.asarray(area).sum()
        n_region_sum = len(cnts)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "{}".format(text), (10, 20),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Camera Feed", frame)
    #cv2.imshow("Thresh", thresh)
    
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        dataFile.close()
        break
    
        # if the 'p' key is pressed, pause recording
        if key == ord("p"):
                monitorPause = not monitorPause
                if monitorPause:
                        text = 'Paused'
                else:
                        text = ''
                        
    # reset reference frame every 1000 samples
    if counter >= 100:
        firstFrame = gray
        counter = 1
    else:
        counter = counter + 1

        #write to file once a minute
        timeDiff = datetime.datetime.now() - initTime
        if timeDiff.seconds >= 60:
                hour_minute = datetime.datetime.now().strftime("%H,%M")
                if monitorPause:
                        dataFile.write(str(hour_minute) + ',' + str(-1) + ',' + str(-1) + '\n')
                else:        
                        dataFile.write(str(hour_minute) + ',' + str(RegionSum/60) + ',' + str(AreaSum/60) + '\n')
                AreaSum = 0
                RegionSum = 0
                initTime = datetime.datetime.now()
        else:
                AreaSum += AreaSum 
                RegionSum += RegionSum
                
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
