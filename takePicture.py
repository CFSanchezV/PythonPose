import mediapipe as mp
import cv2
import numpy as np
import os

#dirname = os.path.dirname(__file__)
#picpath = os.path.join(dirname, 'picturesTaken')

#cam = cv2.VideoCapture(0)

#cv2.namedWindow("test")



def takePictures(ret, frame, picpath, pic_counter):
    if not ret:
        print("failed to grab frame")
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE key press to take pic
        img_name = "opencv_frame_{}.png".format(pic_counter)
        cv2.imwrite(os.path.join(picpath , img_name), frame)
        print("{} written!".format(img_name))

"""
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC key press to close
        print("closing...")
        break
    elif k%256 == 32:
        # SPACE key press to take pic
        img_name = "opencv_frame_{}.png".format(pic_counter)
        cv2.imwrite(os.path.join(picpath , img_name), frame)
        print("{} written!".format(img_name))
        pic_counter += 1

cam.release()

cv2.destroyAllWindows()

"""