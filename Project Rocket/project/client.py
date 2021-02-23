import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylepton import Lepton3
import socket
import base64,zmq
import pickle, struct


def calibrate(x,y):
    return x-22,y-10

def find_max_temp_frame(full_frame, x1,y1,x2,y2):
        max_temp = -273
        min_temp = 655360
        for row in range(y1,y2):
                for col in range(x1,x2):
                    if max_temp<full_frame[row][col]:
                            max_temp = full_frame[row][col]
                    if min_temp>full_frame[row][col]:
                            min_temp = full_frame[row][col]
                
        return round(min_temp,2),round(max_temp,2)
def get_frame():
    context = zmq.Context()
    footage_socket = context.socket(zmq.PUB)
    footage_socket.bind('tcp://*:5555')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            #print(color_image.shape)
            
            image_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            #print(image_gray)
            haar_cascade_face = cv2.CascadeClassifier('/home/pi/Rocket/pyrealsense/data/haarcascades/haarcascade_frontalface_default.xml')
            faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5);
            
            #print('Faces found: ', len(faces_rects))        
            for (x,y,w,h) in faces_rects:
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    #print(x,y,x+w,y+h,w,h)
                    with Lepton3() as L:
                            frame,_ = L.capture()
                            #image = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX)
                            image = cv2.resize(frame, (640, 480))
                            #print(image)
                            x,y = calibrate(x,y)
                            min_frame, max_frame = find_max_temp_frame(image,x,y,x+w,y+h)
                            min_temp, max_temp = min_frame/100 - 273,max_frame/100 -273
                            #print(x,y,x+w,y+h,image.shape)
                            #min_pos_x,min_pos_y,min_pos_temp = find_min_pos(image)
                            cv2.putText(color_image,str(round(max_temp,2))+"C",(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                            #cv2.putText(color_image,str(frame.min()/100 - 273)+"C",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                            
                            
                            """
                            PRE-CALIBRATION
                            image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX)
                
                            np.right_shift(image, 8, image)
                
                            image = cv2.applyColorMap(np.uint8(image), cv2.COLORMAP_JET)
                            
                            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                            faces_rects_lep = haar_cascade_face.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5);
                            for (lx,ly,lw,lh) in faces_rects_lep:
                                cv2.rectangle(image, (lx, ly), (lx+lw, ly+lh), (255, 255, 255), 2)
                                flag = 1
                            cv2.rectangle(image, (x-22, y-10), (x-22+w, y-10+h), (0, 255, 0), 2)
                            cv2.imwrite("lrpton.jpg",image)
                            """
                            
                            """
                            print(frame)
                            print("===================")
                            print(image)
                            print(frame.shape,image.shape,color_image.shape)
                            print(frame.max(),image.max(),frame.min(),image.min())
                            """

            encoded, buffer = cv2.imencode('.jpg', color_image)
            jpg_as_text = base64.b64encode(buffer)
            footage_socket.send(jpg_as_text)
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', color_image)
            #cv2.waitKey(1)
    except Exception as e:
        print(e)
    finally:
        pipeline.stop()

get_frame()
