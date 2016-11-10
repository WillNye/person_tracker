import json
import time
import warnings
import datetime
from collections import deque

from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import cv2

from advanced_detection.tempimage import TempImage


class AdvancedCamShift:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        warnings.filterwarnings("ignore")
        self.conf = json.load(open("conf.json"))
        self.lastUploaded = datetime.datetime.now()
        self.timestamp = datetime.datetime.now()
        self.ts = self.timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        self.pts = deque(maxlen=32)
        (self.dX, self.dY) = (0, 0)
        self.direction = ""
        self.counter = 0
        self.client = None
        self.avg = None
        self.for_show = None
        self.track_windows = None

        # initialize the camera and grab a reference to the raw camera capture
        self.camera = cv2.VideoCapture(0)

        # check to see if the Dropbox should be used
        if self.conf["use_dropbox"]:
            # connect to dropbox and start the session authorization process
            flow = DropboxOAuth2FlowNoRedirect(self.conf["dropbox_key"], self.conf["dropbox_secret"])
            print "[INFO] Authorize this application: {}".format(flow.start())
            authCode = raw_input("Enter auth code here: ").strip()

            # finish the authorization and grab the Dropbox client
            (accessToken, userID) = flow.finish(authCode)
            self.client = DropboxClient(accessToken)
            print "[SUCCESS] dropbox account linked"

    def _define_windows(self, frame):
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
                                                     padding=(8, 8), scale=1.05)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        self.track_windows = rects
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        current_objects = len(pick)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(self.for_show, (xA * 2, yA * 2), (xB * 2, yB * 2), (0, 255, 0), 2)

        if current_objects > 0:
            # draw the text and timestamp on the frame
            self.ts = self.timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(self.for_show, "Occupied by {} people".format(str(current_objects)), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.for_show, self.ts, (10, self.for_show.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
        else:
            cv2.putText(self.for_show, "Unoccupied", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return current_objects

    def follow_objects(self):
        # right now I'm only tracking one object
        person_cords = self.track_windows[0]

        # get the center of the person box
        center = ((person_cords[0] * 2) + person_cords[2], person_cords[1] * 2 + person_cords[3])
        self.pts.appendleft(center)

        # loop over the set of tracked points
        for i in np.arange(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # check to see if enough points have been accumulated in
            # the buffer
            if i == 1 and self.pts[-10] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                self.dX = self.pts[-10][0] - self.pts[i][0]
                self.dY = self.pts[-10][1] - self.pts[i][1]
                (dirX, dirY) = ("", "")

                # ensure there is significant movement in the
                # x-direction
                if np.abs(self.dX) > 20:
                    dirX = "East" if np.sign(self.dX) == 1 else "West"

                # ensure there is significant movement in the
                # y-direction
                if np.abs(self.dY) > 20:
                    dirY = "North" if np.sign(self.dY) == 1 else "South"

                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    self.direction = "{}-{}".format(dirY, dirX)

                # otherwise, only one direction is non-empty
                else:
                    self.direction = dirX if dirX != "" else dirY

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(32 / float(i + 1)) * 1.5)
            # cv2.line(self.for_show, self.pts[i - 1], self.pts[i], (0, 255, 0), thickness)

            # show the movement deltas and the direction of movement on
            # the frame
        cv2.putText(self.for_show, self.direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)
        cv2.putText(self.for_show, "dx: {}, dy: {}".format(self.dX, self.dY),
                    (10, self.for_show.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)

    def start_detection(self):
        # capture frames from the camera
        while True:
            # grab the raw NumPy array representing the image and initialize
            # the timestamp and occupied/unoccupied text
            (grabbed, frame) = self.camera.read()
            self.timestamp = datetime.datetime.now()

            # resize the frame, convert it to grayscale, and blur it
            self.for_show = imutils.resize(frame, width=800)
            frame = imutils.resize(frame, width=400)

            current_objects = self._define_windows(frame)

            if current_objects > 0:
                try:
                    self.follow_objects()
                except Exception as e:
                    print(e)

            # check to see if the room is occupied
            if current_objects > 0 and self.conf['use_dropbox']:
                # check to see if enough time has passed between uploads
                if (self.timestamp - self.lastUploaded).seconds >= self.conf["min_upload_seconds"]:
                    # increment the motion counter
                    self.counter += 1

                    # check to see if the number of frames with consistent motion is
                    # high enough
                    if self.counter >= self.conf["min_motion_frames"]:
                        # check to see if dropbox sohuld be used
                        # write the image to temporary file
                        t = TempImage()
                        cv2.imwrite(t.path, frame)

                        # upload the image to Dropbox and cleanup the tempory image
                        print "[UPLOAD] {}".format(self.ts)
                        path = "{base_path}/{timestamp}.jpg".format(
                            base_path=self.conf["dropbox_base_path"], timestamp=self.ts)
                        self.client.put_file(path, open(t.path, "rb"))
                        t.cleanup()

                        # update the last uploaded timestamp and reset the motion
                        # counter
                        self.lastUploaded = self.timestamp
                else:
                    self.counter = 0

            # otherwise, the room is not occupied
            else:
                self.counter = 0

            # check to see if the frames should be displayed to screen
            if self.conf["show_video"]:
                # display the security feed
                cv2.imshow("Security Feed", self.for_show)
                key = cv2.waitKey(100) & 0xFF

if __name__ == '__main__':
    object_tracing = AdvancedCamShift()
    object_tracing.start_detection()
