
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylepton import Lepton3

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
flag =0 


# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipeline.wait_for_frames()

def calibrate(x,y):
    return x-22,y-10

def find_distance_face(depth_frame, x1,y1,x2,y2):
    """
    for y in range(y1,y2):
        for x in range(x1,x2):
            dist = depth_frame.get_distance(x, y)
    """
    
    depth = np.asanyarray(depth_frame.get_data())
    depth = depth[x1:x2,y1:y2].astype(float)

    # Get data scale from the device and convert to meters
    
    depth = depth * depth_scale
    dist,_,_,_ = cv2.mean(depth)
    #print(dist)
    return round(dist,2)


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
        flag=0
        for (x,y,w,h) in faces_rects:
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #print(x,y,x+w,y+h,w,h)
                
                # Create alignment primitive with color as its target stream:
                """
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
                
                colorizer = rs.colorizer()

                # Update color and depth frames:
                aligned_depth_frame = frames.get_depth_frame()
                #colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

                # Show the two frames together:
                #images = np.hstack((color_image, colorized_depth))
                #plt.imshow(images)
                
                depth = np.asanyarray(aligned_depth_frame.get_data())
                # Crop depth data:
                depth = depth[x:x+w,y:y+h].astype(float)

                # Get data scale from the device and convert to meters
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale
                dist,_,_,_ = cv2.mean(depth)
                print("Detected at {1:.3} meters away.".format(dist))
                """
                
                #depth_frame = frames.get_depth_frame()
                #if not depth_frame: continue
                
                depth =0 #find_distance_face(depth_frame, x,y,x+w,y+h)
                
                cv2.putText(color_image,"{}m away.".format(depth),(x+w+2,y+h+2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                
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
                        cv2.imwrite("depth_temp_op.jpg",color_image)
                        
                        
                        """
                        CALIBRATION
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
        if flag is 1:
            break
		
        """
        Hi SUGI & NIHAL!! (x,y) to (x+w,y+w) is our intended FACE LOCATION (ROI). 
        """
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
finally:
    pipeline.stop()
