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


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

IMAGE_DIR = 'images/'
VIDEO_PATH = 'videos/'

def save_fig(frame):
    # Save the frame as an image
    # the folder has images named 0.png, 1.png, 2.png, ...
    # name it as the next number
    img_name = f"{IMAGE_DIR}{len(os.listdir(IMAGE_DIR))}.png"
    cv2.imwrite(img_name, frame)
    print(f"Image saved as {img_name}")

def test_camera():
    # Initialize video captures for cameras 0, 1, and 2
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)

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

        c = cv2.waitKey(100)
        print(f"c:{c}")

        if c != -1:
            if c == ord('q'):
                break
            elif c == ord('s'):
                save_fig(frame0)
                save_fig(frame1)
                save_fig(frame2)
        
        time1 = time.time()
        
        # print(f"Time taken: {time1 - time0}")

    # Release the video capture objects and close the display window
    cap0.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def process_detection(frame, detector, return_image=True):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(image)

    success = detection_result is not None
    if return_image:
        if detection_result.hand_landmarks:
            annotated_image = draw_landmarks_on_image(frame, detection_result)
        else:
            annotated_image = frame
    else:
        annotated_image = None
        
    return success, detection_result, annotated_image

def handle_camera_detection(cam, detector, return_image=True):
    if not cam.isOpened():
        return False, False, None, None
    
    ret, frame = cam.read()
    if not ret:
        return False, False, None, None
    
    success, detection, annotated_frame = process_detection(frame, detector, return_image=return_image)
    return True, success, detection, annotated_frame
    

def test_hand_landmarker(video_folder_name = None):
    show_image = True
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=5)
    detector = vision.HandLandmarker.create_from_options(options)
    
    if video_folder_name is None:
        # Initialize video captures for each camera
        cap0 = cv2.VideoCapture(0)  # Camera 0
        cap1 = cv2.VideoCapture(1)  # Camera 1
        cap2 = cv2.VideoCapture(2)  # Camera 2
    else:
        # Initialize video captures for cameras 0, 1, and 2
        cap0 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/l.avi")
        cap1 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/m.avi")
        cap2 = cv2.VideoCapture(f"{VIDEO_PATH}{video_folder_name}/r.avi")

    with ThreadPoolExecutor(max_workers=3) as executor:
        while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
            # ret0, frame0 = cap0.read()
            # ret1, frame1 = cap1.read()
            # ret2, frame2 = cap2.read()
            
            time0 = time.time()
            
            future0 = executor.submit(handle_camera_detection, cap0, detector, show_image)
            future1 = executor.submit(handle_camera_detection, cap1, detector, show_image)
            future2 = executor.submit(handle_camera_detection, cap2, detector, show_image)
            
            ret0, success0, result0, annotated_frame0 = future0.result()
            ret1, success1, result1, annotated_frame1 = future1.result()
            ret2, success2, result2, annotated_frame2 = future2.result()
            
            if not ret0 or not ret1 or not ret2:
                break
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                continue    

            # success0, annotated_frame0 = process_detection(frame0, detector)
            # success1, annotated_frame1 = process_detection(frame1, detector)
            # success2, annotated_frame2 = process_detection(frame2, detector)
            
            # success0, annotated_frame0 = future0.result()
            # success1, annotated_frame1 = future1.result()
            # success2, annotated_frame2 = future2.result()
            
            # height = min(annotated_frame0.shape[0], annotated_frame1.shape[0], annotated_frame2.shape[0])
            # annotated_frame0 = cv2.resize(annotated_frame0, (int(annotated_frame0.shape[1] * height / annotated_frame0.shape[0]), height))
            # annotated_frame1 = cv2.resize(annotated_frame1, (int(annotated_frame1.shape[1] * height / annotated_frame1.shape[0]), height))
            # annotated_frame2 = cv2.resize(annotated_frame2, (int(annotated_frame2.shape[1] * height / annotated_frame2.shape[0]), height))
            
            if show_image:
                concatenated_frame = np.hstack((annotated_frame0, annotated_frame1, annotated_frame2))
                cv2.imshow('MediaPipe Hand Landmarks', concatenated_frame)
            
            time1 = time.time()
            
            print(f"Time taken: {time1 - time0}")
        
        cap0.release()
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    test_hand_landmarker('1716957191.323741')