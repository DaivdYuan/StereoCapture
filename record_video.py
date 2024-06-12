# install model bundle under the current directory
# wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ThreadPoolExecutor

VIDEO_PATH = 'videos/'

def record_camera():
    # Initialize video captures for cameras 0, 1, and 2
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)
    
    cur_time = time.time()
    
    # current video path is current time
    video_path = f"{VIDEO_PATH}{cur_time}/"
    
    # create the video path
    os.makedirs(video_path, exist_ok=True)
    
    # create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out0 = cv2.VideoWriter(f"{video_path}l.avi", fourcc, 20.0, (int(cap0.get(3)), int(cap0.get(4))))
    out1 = cv2.VideoWriter(f"{video_path}m.avi", fourcc, 20.0, (int(cap1.get(3)), int(cap1.get(4))))
    out2 = cv2.VideoWriter(f"{video_path}r.avi", fourcc, 20.0, (int(cap2.get(3)), int(cap2.get(4))))

    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
        time0 = time.time()
        
        # Read frames from each camera
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or not ret1 or not ret2:
            break

        # Resize frames to the same height (if necessary)
        # height = min(frame0.shape[0], frame1.shape[0], frame2.shape[0])
        # frame0 = cv2.resize(frame0, (int(frame0.shape[1] * height / frame0.shape[0]), height))
        # frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
        # frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))
        
        # # resize the frames to 640x480
        # frame0 = cv2.resize(frame0, (640, 480))
        # frame1 = cv2.resize(frame1, (640, 480))
        # frame2 = cv2.resize(frame2, (640, 480))
        
        # write the frames to the video
        out0.write(frame0)
        out1.write(frame1)
        out2.write(frame2)
        
        # Concatenate frames horizontally
        concatenated_frame = np.hstack((frame0, frame1, frame2))

        # Show the concatenated frame
        cv2.imshow('Camera Stream', concatenated_frame)
        
        c = cv2.waitKey(1)
        print(f"c:{c}")

        if c != -1:
            if c == ord('q'):
                break

        time1 = time.time()
        
        print(f"Time taken: {time1 - time0}")

    # Release the video capture objects and close the display window
    cap0.release()
    cap1.release()
    cap2.release()
    
    out0.release()
    out1.release()
    out2.release()
    
    cv2.destroyAllWindows()
    
    return cur_time

def play_video(video_folder_name):
    # Initialize video captures for cameras 0, 1, and 2
    cap0 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/l.avi")
    cap1 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/m.avi")
    cap2 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/r.avi")
    
    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
        time0 = time.time()
        
        # Read frames from each camera
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or not ret1 or not ret2:
            break

        # Resize frames to the same height (if necessary)
        height = min(frame0.shape[0], frame1.shape[0], frame2.shape[0])
        frame0 = cv2.resize(frame0, (int(frame0.shape[1] * height / frame0.shape[0]), height))
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))
        
        # Concatenate frames horizontally
        concatenated_frame = np.hstack((frame0, frame1, frame2))

        # Show the concatenated frame
        cv2.imshow('Camera Stream', concatenated_frame)
        
        c = cv2.waitKey(1)
        print(f"c:{c}")

        if c != -1:
            if c == ord('q'):
                break

        time1 = time.time()
        
        print(f"Time taken: {time1 - time0}")

    # Release the video capture objects and close the display window
    cap0.release()
    cap1.release()
    cap2.release()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #record_camera()
    play_video('1716957191.323741')