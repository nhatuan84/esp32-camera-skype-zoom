import socket
import os
import fcntl
import numpy as np
import cv2
from v4l2 import *
import time

ESP32_SERVER_IP = '192.168.1.3' 
PORT = 8088
VID_WIDTH = 320
VID_HEIGHT = 240
vd = os.open('/dev/video6', os.O_RDWR)
fmt = v4l2_format()
fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
fcntl.ioctl(vd, VIDIOC_G_FMT, fmt)
fmt.fmt.pix.width = VID_WIDTH
fmt.fmt.pix.height = VID_HEIGHT
fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420
fmt.fmt.pix.sizeimage = VID_WIDTH * VID_HEIGHT * 3
fmt.fmt.pix.field = V4L2_FIELD_NONE
fcntl.ioctl(vd, VIDIOC_S_FMT, fmt)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

while True:
    buffer = bytearray()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ESP32_SERVER_IP, PORT))
        while True:
            data = s.recv(1024)
            len = data.find(b'\r\n')
            if(len > 0):
                buffer += data[0:len]
                image_bytes = np.frombuffer(buffer, dtype=np.uint8) 
                frame = cv2.imdecode(image_bytes, flags=cv2.IMREAD_COLOR)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.equalizeHist(frame_gray)
                #-- Detect faces
                faces = face_cascade.detectMultiScale(frame_gray)
                for (x,y,w,h) in faces:
                    center = (x + w//2, y + h//2)
                    faceROI = frame_gray[y:y+h,x:x+w]
                    #-- In each face, detect eyes
                    eyes = eyes_cascade.detectMultiScale(faceROI)
                    for (x2,y2,w2,h2) in eyes:
                        eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                        radius = int(round((w2 + h2)*0.25))
                        frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), -1)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
                os.write(vd, frame)
                s.close()
                break
            else:
                buffer += data
