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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from particle_filter import ParticleFilter3D
from camera_test import handle_camera_detection

from utils import get_ray, get_least_square_point


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

IMAGE_DIR = 'images/'
VIDEO_DIR = 'videos/'
LANDMARK_DIR = 'landmarks/'

class HandLandmarkerRecorder:
    def __init__(self, output_path):
        self.output_path = output_path
        self.reset()
        
    def reset(self):
        self.mask_left = []
        self.mask_right = []
        self.landmarks_left = []
        self.landmarks_right = []
        
    def add_frame(self, detection_result):
        has_left = 0
        has_right = 0
        left_landmarks = np.zeros((21, 3))
        right_landmarks = np.zeros((21, 3))
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            
            landmarks = np.array([np.array([landmark.x, landmark.y, landmark.z]) for landmark in hand_landmarks])
            assert landmarks.shape == (21, 3)
            
            if handedness[0].category_name == 'Left':
                has_left = 1
                left_landmarks = landmarks
            elif handedness[0].category_name == 'Right':
                has_right = 1
                right_landmarks = landmarks
                
        self.mask_left.append(has_left)
        self.mask_right.append(has_right)
        self.landmarks_left.append(left_landmarks)
        self.landmarks_right.append(right_landmarks)   
    
    def save(self):
        np.savez(self.output_path, mask_left=self.mask_left, mask_right=self.mask_right, landmarks_left=self.landmarks_left, landmarks_right=self.landmarks_right)
        print(f"Data saved as {self.output_path}")
        self.reset()
        
    def load(self):
        try:
            data = np.load(self.output_path)
            self.mask_left = data['mask_left']
            self.mask_right = data['mask_right']
            self.landmarks_left = data['landmarks_left']
            self.landmarks_right = data['landmarks_right']
            print(f"Data loaded from {self.output_path}")
        except:
            print(f"Data not found in {self.output_path}")
            return None
        
    def get(self, idx):
        return self.mask_left[idx], self.mask_right[idx], self.landmarks_left[idx], self.landmarks_right[idx]

def plot_point(frames, mtx, rs, ts, point, color = (0, 255, 0)):
    for i, (r, t, frame) in enumerate(zip(rs, ts, frames)):
        imagepoint = np.dot(mtx, np.dot(r, point) + t)
        imagepoint /= imagepoint[2]
        cv2.circle(frame, tuple(imagepoint[0:2, 0].astype(int)), 5, color, -1)

def plot_line(frames, mtx, rs, ts, r, v, color = (0, 255, 0)):
    pass

def do_hand_landmarker(video_folder_name, overwrite=False):
    show_image = True
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=5)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # create the landmark path
    landmark_path = f"{LANDMARK_DIR}{video_folder_name}/"
    os.makedirs(landmark_path, exist_ok=True)
    
    if not overwrite:
        if os.path.exists(f"{landmark_path}l.npz") and os.path.exists(f"{landmark_path}m.npz") and os.path.exists(f"{landmark_path}r.npz"):
            print("Landmark files already exist. Skipping the detection process.")
            return
    
    landmark_recorder_l = HandLandmarkerRecorder(f"{landmark_path}l.npz")
    landmark_recorder_m = HandLandmarkerRecorder(f"{landmark_path}m.npz")
    landmark_recorder_r = HandLandmarkerRecorder(f"{landmark_path}r.npz")
    
    if video_folder_name is None:
        # Initialize video captures for each camera
        # cap0 = cv2.VideoCapture(0)  # Camera 0
        # cap1 = cv2.VideoCapture(1)  # Camera 1
        # cap2 = cv2.VideoCapture(2)  # Camera 2
        raise ValueError("Please provide a video folder name")
    else:
        # Initialize video captures for cameras 0, 1, and 2
        cap0 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/l.avi")
        cap1 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/m.avi")
        cap2 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/r.avi")

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
            
            # save the landmarks
            landmark_recorder_l.add_frame(result0)
            landmark_recorder_m.add_frame(result1)
            landmark_recorder_r.add_frame(result2)

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
        
        plt.close()

        landmark_recorder_l.save()
        landmark_recorder_m.save()
        landmark_recorder_r.save()
        
def do_pf_fitting_on_image(video_folder_name):
    # Load the calibration results
    calibration = np.load('calibration/calibration.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    r_vec_l = calibration['r_vec_l']
    t_vec_l = calibration['t_vec_l']
    r_vec_r = calibration['r_vec_r']
    t_vec_r = calibration['t_vec_r']
    r_vec_m = calibration['r_vec_m']
    t_vec_m = calibration['t_vec_m']
    
    # pf
    num_particles = 200000
    val_range = [12, 12, 30]
    diffusion_rate = 1.0
    random_resample_rate = 0.05
    retry = 5
    base_value = 1e-3

    pf = ParticleFilter3D(num_particles, val_range, diffusion_rate, random_resample_rate, base_value, retry)
    pf.init_camera(mtx, dist, r_vec_l, t_vec_l, r_vec_m, t_vec_m, r_vec_r, t_vec_r)
    cov_eff = 1
    covs = {
        'l': np.array([cov_eff, cov_eff]),
        'm': np.array([cov_eff, cov_eff]),
        'r': np.array([cov_eff, cov_eff])
    }
    
    landmark_path = f"{LANDMARK_DIR}{video_folder_name}/"    
    landmark_recorder_l = HandLandmarkerRecorder(f"{landmark_path}l.npz")
    landmark_recorder_m = HandLandmarkerRecorder(f"{landmark_path}m.npz")
    landmark_recorder_r = HandLandmarkerRecorder(f"{landmark_path}r.npz")
    landmark_recorder_l.load()
    landmark_recorder_m.load()
    landmark_recorder_r.load()
    
    cap0 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/l.avi")
    cap1 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/m.avi")
    cap2 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/r.avi")
    
    h = 720
    w = 1280

    cnt = 0

    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
        
        time0 = time.time()
        
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or not ret1 or not ret2:
            break
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue    
        
        # plot the first point from the recorder
        has_left_l, _, detection_left_l, _ = landmark_recorder_l.get(cnt)
        has_left_m, _, detection_left_m, _ = landmark_recorder_m.get(cnt)
        has_left_r, _, detection_left_r, _ = landmark_recorder_r.get(cnt)
        
        means = {}
        
        if has_left_l:
            #print((int(detection_left_l[0][0] * h), int(detection_left_l[0][1] * w)))
            point = np.array([int(detection_left_l[0][0] * w), int(detection_left_l[0][1] * h)])
            cv2.circle(frame0, point, 5, (0, 255, 0), -1)
            means['l'] = point
        if has_left_m:
            point = np.array([int(detection_left_m[0][0] * w), int(detection_left_m[0][1] * h)])
            cv2.circle(frame1, point, 5, (0, 255, 0), -1)
            means['m'] = point
        if has_left_r:
            point = np.array([int(detection_left_r[0][0] * w), int(detection_left_r[0][1] * h)])
            cv2.circle(frame2, point, 5, (0, 255, 0), -1)
            means['r'] = point

        top_points, variance = pf.step_filter_2d(means, covs)
        estimated_points = top_points
        
        imageestimated_points_l, _ = cv2.projectPoints(estimated_points, r_vec_l, t_vec_l, mtx, dist)
        imageestimated_points_m, _ = cv2.projectPoints(estimated_points, r_vec_m, t_vec_m, mtx, dist)
        imageestimated_points_r, _ = cv2.projectPoints(estimated_points, r_vec_r, t_vec_r, mtx, dist)
        
        for i in range(9, -1, -1):
            if i == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255) 
            try:   
                cv2.circle(frame0, tuple(imageestimated_points_l[i].ravel().astype(int)), 5, color, -1)
                cv2.circle(frame1, tuple(imageestimated_points_m[i].ravel().astype(int)), 5, color, -1)        
                cv2.circle(frame2, tuple(imageestimated_points_r[i].ravel().astype(int)), 5, color, -1)
            except:
                pass
        
        # concatenated_frame = np.hstack((annotated_frame0, annotated_frame1, annotated_frame2))
        concatenated_frame = np.hstack((frame0, frame1, frame2))
        cv2.imshow('MediaPipe Hand Landmarks', concatenated_frame)
        c = cv2.waitKey(1)
        
        time1 = time.time()
        
        print(f"Time taken: {time1 - time0}")

        cnt += 1
    
    cap0.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    plt.close()

        
def show_landmarks(video_folder_name):
    # Load the calibration results
    calibration = np.load('calibration/calibration.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    r_vec_l = calibration['r_vec_l']
    t_vec_l = calibration['t_vec_l']
    r_vec_r = calibration['r_vec_r']
    t_vec_r = calibration['t_vec_r']
    r_vec_m = calibration['r_vec_m']
    t_vec_m = calibration['t_vec_m']
    
    r_mat_l, _ = cv2.Rodrigues(r_vec_l)
    r_mat_r, _ = cv2.Rodrigues(r_vec_r)
    r_mat_m, _ = cv2.Rodrigues(r_vec_m)
    
    landmark_path = f"{LANDMARK_DIR}{video_folder_name}/"    
    landmark_recorder_l = HandLandmarkerRecorder(f"{landmark_path}l.npz")
    landmark_recorder_m = HandLandmarkerRecorder(f"{landmark_path}m.npz")
    landmark_recorder_r = HandLandmarkerRecorder(f"{landmark_path}r.npz")
    landmark_recorder_l.load()
    landmark_recorder_m.load()
    landmark_recorder_r.load()
    
    cap0 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/l.avi")
    cap1 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/m.avi")
    cap2 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/r.avi")
    
    h = 720
    w = 1280

    cnt = 0

    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
        
        time0 = time.time()
        
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or not ret1 or not ret2:
            break
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue    
        
        # plot the first point from the recorder
        has_left_l, _, detection_left_l, _ = landmark_recorder_l.get(cnt)
        has_left_m, _, detection_left_m, _ = landmark_recorder_m.get(cnt)
        has_left_r, _, detection_left_r, _ = landmark_recorder_r.get(cnt)
        
        pts = []
        vecs = []
        if has_left_l:
            #print((int(detection_left_l[0][0] * h), int(detection_left_l[0][1] * w)))
            point = np.array([int(detection_left_l[0][0] * w), int(detection_left_l[0][1] * h)])
            cv2.circle(frame0, point, 5, (0, 255, 0), -1)
            
            point = np.concatenate((point, np.array([1]))).reshape(-1, 1)
            cam_ray, cam_center = get_ray(mtx, r_mat_l, t_vec_l, point)
            pts.append(cam_center)
            vecs.append(cam_ray)
            # for i in range(1, 30):
            #     point = cam_center + cam_ray * i
                
            #     imagepoint_l = np.dot(mtx, np.dot(r_mat_l, point) + t_vec_l)
            #     imagepoint_l /= imagepoint_l[2]
            #     imagepoint_r = np.dot(mtx, np.dot(r_mat_r, point) + t_vec_r)
            #     imagepoint_r /= imagepoint_r[2]
            #     imagepoint_m = np.dot(mtx, np.dot(r_mat_m, point) + t_vec_m)
            #     imagepoint_m /= imagepoint_m[2]
                                
            #     # draw the projected points on the frames
            #     cv2.circle(frame0, tuple(imagepoint_l[0:2, 0].astype(int)), 5, (0, 255, 255), -1)
            #     cv2.circle(frame1, tuple(imagepoint_m[0:2, 0].astype(int)), 5, (0, 255, 255), -1)
            #     cv2.circle(frame2, tuple(imagepoint_r[0:2, 0].astype(int)), 5, (0, 255, 255), -1)
        if has_left_m:
            point = np.array([int(detection_left_m[0][0] * w), int(detection_left_m[0][1] * h)])
            cv2.circle(frame1, point, 5, (0, 255, 0), -1)
            
            point = np.concatenate((point, np.array([1]))).reshape(-1, 1)
            cam_ray, cam_center = get_ray(mtx, r_mat_m, t_vec_m, point)
            pts.append(cam_center)
            vecs.append(cam_ray)
            # for i in range(1, 30):
            #     point = cam_center + cam_ray * i
                
            #     imagepoint_l = np.dot(mtx, np.dot(r_mat_l, point) + t_vec_l)
            #     imagepoint_l /= imagepoint_l[2]
            #     imagepoint_r = np.dot(mtx, np.dot(r_mat_r, point) + t_vec_r)
            #     imagepoint_r /= imagepoint_r[2]
            #     imagepoint_m = np.dot(mtx, np.dot(r_mat_m, point) + t_vec_m)
            #     imagepoint_m /= imagepoint_m[2]
                                
            #     # draw the projected points on the frames
            #     cv2.circle(frame0, tuple(imagepoint_l[0:2, 0].astype(int)), 5, (255, 0, 255), -1)
            #     cv2.circle(frame1, tuple(imagepoint_m[0:2, 0].astype(int)), 5, (255, 0, 255), -1)
            #     cv2.circle(frame2, tuple(imagepoint_r[0:2, 0].astype(int)), 5, (255, 0, 255), -1)
                
        if has_left_r:
            point = np.array([int(detection_left_r[0][0] * w), int(detection_left_r[0][1] * h)])
            cv2.circle(frame2, point, 5, (0, 255, 0), -1)
            
            point = np.concatenate((point, np.array([1]))).reshape(-1, 1)
            cam_ray, cam_center = get_ray(mtx, r_mat_r, t_vec_r, point)
            pts.append(cam_center)
            vecs.append(cam_ray)
            # for i in range(1, 30):
            #     point = cam_center + cam_ray * i
                
            #     imagepoint_l = np.dot(mtx, np.dot(r_mat_l, point) + t_vec_l)
            #     imagepoint_l /= imagepoint_l[2]
            #     imagepoint_r = np.dot(mtx, np.dot(r_mat_r, point) + t_vec_r)
            #     imagepoint_r /= imagepoint_r[2]
            #     imagepoint_m = np.dot(mtx, np.dot(r_mat_m, point) + t_vec_m)
            #     imagepoint_m /= imagepoint_m[2]
                                
            #     # draw the projected points on the frames
            #     cv2.circle(frame0, tuple(imagepoint_l[0:2, 0].astype(int)), 5, (255, 255, 0), -1)
            #     cv2.circle(frame1, tuple(imagepoint_m[0:2, 0].astype(int)), 5, (255, 255, 0), -1)
            #     cv2.circle(frame2, tuple(imagepoint_r[0:2, 0].astype(int)), 5, (255, 255, 0), -1)
        
        pts = np.array(pts).reshape(-1, 3)
        vecs = np.array(vecs).reshape(-1, 3)
        
        least_square_point = get_least_square_point(pts, vecs)
        
        if least_square_point is not None:
            imagepoint_l = np.dot(mtx, np.dot(r_mat_l, least_square_point) + t_vec_l)
            imagepoint_l /= imagepoint_l[2]
            imagepoint_r = np.dot(mtx, np.dot(r_mat_r, least_square_point) + t_vec_r)
            imagepoint_r /= imagepoint_r[2]
            imagepoint_m = np.dot(mtx, np.dot(r_mat_m, least_square_point) + t_vec_m)
            imagepoint_m /= imagepoint_m[2]
            
            cv2.circle(frame0, tuple(imagepoint_l[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(frame1, tuple(imagepoint_m[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(frame2, tuple(imagepoint_r[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
        
        
        
        
        
        # concatenated_frame = np.hstack((annotated_frame0, annotated_frame1, annotated_frame2))
        concatenated_frame = np.hstack((frame0, frame1, frame2))
        cv2.imshow('MediaPipe Hand Landmarks', concatenated_frame)
        c = cv2.waitKey(1)
        
        time1 = time.time()
        
        print(f"Time taken: {time1 - time0}")

        cnt += 1
    
    cap0.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def do_line_fitting_on_image(video_folder_name):
    do_hand_landmarker(video_folder_name)
    
    # Load the calibration results
    calibration = np.load('calibration/calibration.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']
    r_vec_l = calibration['r_vec_l']
    t_vec_l = calibration['t_vec_l']
    r_vec_r = calibration['r_vec_r']
    t_vec_r = calibration['t_vec_r']
    r_vec_m = calibration['r_vec_m']
    t_vec_m = calibration['t_vec_m']
    
    r_mat_l, _ = cv2.Rodrigues(r_vec_l)
    r_mat_r, _ = cv2.Rodrigues(r_vec_r)
    r_mat_m, _ = cv2.Rodrigues(r_vec_m)
    
    Rs = [r_mat_l, r_mat_m, r_mat_r]
    ts = [t_vec_l, t_vec_m, t_vec_r]
    
    landmark_path = f"{LANDMARK_DIR}{video_folder_name}/"    
    landmark_recorder_l = HandLandmarkerRecorder(f"{landmark_path}l.npz")
    landmark_recorder_m = HandLandmarkerRecorder(f"{landmark_path}m.npz")
    landmark_recorder_r = HandLandmarkerRecorder(f"{landmark_path}r.npz")
    landmark_recorder_l.load()
    landmark_recorder_m.load()
    landmark_recorder_r.load()
    
    cap0 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/l.avi")
    cap1 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/m.avi")
    cap2 = cv2.VideoCapture(f"{VIDEO_DIR}{video_folder_name}/r.avi")
    
    # Enable interactive mode
    plt.ion()

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # ax limits
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 5])
    
    h = 720
    w = 1280

    cnt = 0

    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened():
        time0 = time.time()
        
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret0 or not ret1 or not ret2:
            break
        
        frames = [frame0, frame1, frame2]
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue    
        
        # plot the first point from the recorder
        has_left_l, has_right_l, detection_left_l, detection_right_l = landmark_recorder_l.get(cnt)
        has_left_m, has_right_m, detection_left_m, detection_right_m = landmark_recorder_m.get(cnt)
        has_left_r, has_right_r, detection_left_r, detection_right_r = landmark_recorder_r.get(cnt)
        
        has_left = [has_left_l, has_left_m, has_left_r]
        has_right = [has_right_l, has_right_m, has_right_r]
        detections_left = [detection_left_l, detection_left_m, detection_left_r]
        detections_right = [detection_right_l, detection_right_m, detection_right_r]
        
        left_idx = []
        right_idx = []
        for i in range(3):
            if has_left[i]:
                left_idx.append(i)
            if has_right[i]:
                right_idx.append(i)
        
        cur_left_pts = np.zeros([21, 3, 1])
        cur_right_pts = np.zeros([21, 3, 1])
        
        if len(left_idx) >= 2:
            ray_vecs = np.zeros([21, len(left_idx), 3])
            ray_points = np.zeros([21, len(left_idx), 3])
            for i, cam in enumerate(left_idx):  # cam
                for j in range(21):  # key point
                    p = detections_left[cam][j]
                    point = np.array([int(p[0] * w), int(p[1] * h), 1])
                    
                    # circle the point
                    # cv2.circle(frames[cam], tuple(point[0:2]), 5, (0, 255, 255), -1)
                    
                    v, p = get_ray(mtx, Rs[cam], ts[cam], point)
                                        
                    ray_vecs[j, i, :] = v.reshape(-1)
                    ray_points[j, i, :] = p.reshape(-1)
                    
            for j in range(21):
                vecs = ray_vecs[j, :, :]
                points = ray_points[j, :, :]
                least_square_point = get_least_square_point(points, vecs)
                
                # print(f"Least square point: {least_square_point}")
                
                cur_left_pts[j, :, :] = least_square_point
                
                # plot the point
                plot_point(frames, mtx, Rs, ts, least_square_point, (0, 255, 0))
                
            # ax.scatter(cur_left_pts[:, 0, 0], cur_left_pts[:, 2, 0], -cur_left_pts[:, 1, 0], c='r', marker='o', label='Left Points')
                
        if len(right_idx) >= 2:
            ray_vecs = np.zeros([21, len(right_idx), 3])
            ray_points = np.zeros([21, len(right_idx), 3])
            for i, cam in enumerate(right_idx):
                for j in range(21):
                    p = detections_right[cam][j]
                    point = np.array([int(p[0] * w), int(p[1] * h), 1])
                    
                    # circle the point
                    # cv2.circle(frames[cam], tuple(point[0:2]), 5, (0, 255, 255), -1)
                    
                    v, p = get_ray(mtx, Rs[cam], ts[cam], point)
                                        
                    ray_vecs[j, i, :] = v.reshape(-1)
                    ray_points[j, i, :] = p.reshape(-1)
                    
            for j in range(21):
                vecs = ray_vecs[j, :, :]
                points = ray_points[j, :, :]
                least_square_point = get_least_square_point(points, vecs)
                
                # print(f"Least square point: {least_square_point}")
                
                cur_right_pts[j, :, :] = least_square_point
                
                # plot the point
                plot_point(frames, mtx, Rs, ts, least_square_point, (0, 255, 255))

            # ax.scatter(cur_right_pts[:, 0, 0], cur_right_pts[:, 2, 0], -cur_right_pts[:, 1, 0], c='b', marker='o', label='Right Points')
        
        concatenated_frame = np.hstack((frame0, frame1, frame2))
        cv2.imshow('MediaPipe Hand Landmarks', concatenated_frame)
        c = cv2.waitKey(1)
        
        # Clear the previous plot
        ax.cla()
        
        # ax limits
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-5, 5])
        
        if len(left_idx) >= 2:
            ax.scatter(cur_left_pts[:, 0, 0], cur_left_pts[:, 2, 0], -cur_left_pts[:, 1, 0], c='r', marker='o', label='Left Points')
            
        if len(right_idx) >= 2:
            ax.scatter(cur_right_pts[:, 0, 0], cur_right_pts[:, 2, 0], -cur_right_pts[:, 1, 0], c='b', marker='o', label='Right Points')
        
        # plt.show()
        plt.pause(0.05)
        
        time1 = time.time()
        
        print(f"Time taken: {time1 - time0}")

        cnt += 1
    
    cap0.release()
    cap1.release()
    cap2.release()
    
    plt.ioff()
    
    cv2.destroyAllWindows()
                
        
    
    
if __name__ == '__main__':
    #do_hand_landmarker('1716957191.323741', True)
    #do_pf_fitting_on_image('1716957191.323741')
    #show_landmarks('1716957191.323741')
    do_line_fitting_on_image('1716957191.323741')
    #do_hand_landmarker('1717221217.788768', True)
    #do_line_fitting_on_image('1717221217.788768')
    
    pass