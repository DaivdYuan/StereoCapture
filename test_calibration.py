import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
from scipy.linalg import null_space

from particle_filter import ParticleFilter3D
from utils import get_ray, get_least_square_point

# np.savez('calibration/calibration.npz', mtx=mtx, dist=dist, r_vec_l=r_vec_l, t_vec_l=t_vec_l, r_vec_r=r_vec_r, t_vec_r=t_vec_r, r_vec_m=r_vec_m, t_vec_m=t_vec_m)

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

test_3d_points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.float32)
num_points = test_3d_points.shape[0]

max_steps = 20

# data
noise = 10.0

error_3_total = 0
errot_3_cnt = 0
error_2_total = 0
error_2_cnt = 0
error_pf_total = 0
error_pf_cnt = 0

# pf
num_particles = 200000
val_range = [10, 10, 20]
diffusion_rate = 0.5
random_resample_rate = 0.2
base_value = 1e-4

pf = ParticleFilter3D(num_particles, val_range, diffusion_rate, random_resample_rate, base_value)
pf.init_camera(mtx, dist, r_vec_l, t_vec_l, r_vec_m, t_vec_m, r_vec_r, t_vec_r)
cov_eff = 0.5
covs = {
    'l': np.array([cov_eff, cov_eff]),
    'm': np.array([cov_eff, cov_eff]),
    'r': np.array([cov_eff, cov_eff])
}
    

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

step_cnt = 0
while cap0.isOpened() and cap1.isOpened() and cap2.isOpened() and step_cnt < max_steps:
    step_cnt += 1
    time0 = time.time()
    
    # Read frames from each camera
    ret0, frame0 = cap0.read() # l
    ret1, frame1 = cap1.read() # m
    ret2, frame2 = cap2.read() # r
    
    if not ret0 or not ret1 or not ret2:
        break
    
    # draw the text 'l', 'm', 'r' on the frames
    cv2.putText(frame0, 'l', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame1, 'm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame2, 'r', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    
    # project the 3D points to the image planes
    imagepoints_l, _ = cv2.projectPoints(test_3d_points, r_vec_l, t_vec_l, mtx, dist)
    imagepoints_m, _ = cv2.projectPoints(test_3d_points, r_vec_m, t_vec_m, mtx, dist)
    imagepoints_r, _ = cv2.projectPoints(test_3d_points, r_vec_r, t_vec_r, mtx, dist)
    
    noise_l = np.random.randn(num_points, 1,  2) * noise
    noise_m = np.random.randn(num_points, 1,  2) * noise
    noise_r = np.random.randn(num_points, 1, 2) * noise
    
    imagepoints_l += noise_l
    imagepoints_m += noise_m
    imagepoints_r += noise_r
    
    if True:
        point0 = test_3d_points[0].reshape(-1, 1)
        
        r_mat_l = Rotation.from_rotvec(r_vec_l.reshape(3,)).as_matrix()
        r_mat_m = Rotation.from_rotvec(r_vec_m.reshape(3,)).as_matrix()
        r_mat_r = Rotation.from_rotvec(r_vec_r.reshape(3,)).as_matrix()
        
        # project the 3D point to the image planes by matrix multiplication
        imagepoint_l0 = imagepoints_l[0].ravel().reshape(-1, 1)
        imagepoint_m0 = imagepoints_m[0].ravel().reshape(-1, 1)
        imagepoint_r0 = imagepoints_r[0].ravel().reshape(-1, 1)
                        
        # draw the projected points on the frames
        cv2.circle(frame0, tuple(imagepoint_l0[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame1, tuple(imagepoint_m0[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame2, tuple(imagepoint_r0[0:2, 0].astype(int)), 5, (0, 0, 255), -1)
        
        ray_vector_l, cam_center_l = get_ray(mtx, r_mat_l, t_vec_l, imagepoint_l0)
        
        # for i in range(1, 30):
        #     point = cam_center_l + ray_vector_l * i
            
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
            
        ray_vector_m, cam_center_m = get_ray(mtx, r_mat_m, t_vec_m, imagepoint_m0)
        
        # for i in range(1, 30):
        #     point = cam_center_m + ray_vector_m * i
            
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
            
        ray_vector_r, cam_center_r = get_ray(mtx, r_mat_r, t_vec_r, imagepoint_r0)
        
        # for i in range(1, 30):
        #     point = cam_center_r + ray_vector_r * i
            
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
        
        vecs = np.stack([ray_vector_l, ray_vector_m, ray_vector_r]).reshape(3, 3)
        pts = np.stack([cam_center_l, cam_center_m, cam_center_r]).reshape(3, 3)
        least_square_point = get_least_square_point(pts, vecs)
        
        # print(f"Least square point: {least_square_point}")
        err_3 = np.linalg.norm(least_square_point - point0)
        print(f"Error 3 line: {err_3}")
        
        error_3_total += err_3
        errot_3_cnt += 1
        
        # test interection using only 2 lines
        for i in range(3):
            tmp_vecs = np.delete(vecs, i, axis=0)
            tmp_pts = np.delete(pts, i, axis=0)
            tmp_point = get_least_square_point(tmp_pts, tmp_vecs)
            err_2 = np.linalg.norm(tmp_point - point0)
            print(f"Error 2 line: {err_2}")
            error_2_total += err_2
            error_2_cnt += 1
        
        # project the least square point to the image planes
        imagepoint_l = np.dot(mtx, np.dot(r_mat_l, least_square_point) + t_vec_l)
        imagepoint_l /= imagepoint_l[2]
        imagepoint_r = np.dot(mtx, np.dot(r_mat_r, least_square_point) + t_vec_r)
        imagepoint_r /= imagepoint_r[2]
        imagepoint_m = np.dot(mtx, np.dot(r_mat_m, least_square_point) + t_vec_m)
        imagepoint_m /= imagepoint_m[2]
        
        # draw the projected points on the frames
        cv2.circle(frame0, tuple(imagepoint_l[0:2, 0].astype(int)), 5, (255, 255, 255), -1)
        cv2.circle(frame1, tuple(imagepoint_m[0:2, 0].astype(int)), 5, (255, 255, 255), -1)
        cv2.circle(frame2, tuple(imagepoint_r[0:2, 0].astype(int)), 5, (255, 255, 255), -1)
    
    
    if True:
        # test the particle filter
        means = {
            'l': imagepoints_l[0].ravel(),
            'm': imagepoints_m[0].ravel(),
            'r': imagepoints_r[0].ravel()
        }
        pf.step_filter_2d(means, covs)
        top_points = pf.get_top_particles(20)
        variance = np.var(top_points, axis=0)
        
        #avg_estimation = pf.get_estimate(num_top_particles=3)
        # avg_estimation = top_points[0]
        # estimated_points = np.concatenate((np.array([avg_estimation]), top_points))
        estimated_points = top_points
        err = np.linalg.norm(estimated_points[0] - test_3d_points[0])
        print(f"Error pf: {err}")
        
        if step_cnt >= 2:
            error_pf_total += err
            error_pf_cnt += 1
        
        imageestimated_points_l, _ = cv2.projectPoints(estimated_points, r_vec_l, t_vec_l, mtx, dist)
        imageestimated_points_m, _ = cv2.projectPoints(estimated_points, r_vec_m, t_vec_m, mtx, dist)
        imageestimated_points_r, _ = cv2.projectPoints(estimated_points, r_vec_r, t_vec_r, mtx, dist)
        
        for i in range(10):
            if i == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)    
            cv2.circle(frame0, tuple(imageestimated_points_l[i].ravel().astype(int)), 5, color, -1)
            cv2.circle(frame1, tuple(imageestimated_points_m[i].ravel().astype(int)), 5, color, -1)        
            cv2.circle(frame2, tuple(imageestimated_points_r[i].ravel().astype(int)), 5, color, -1)
    
    # draw the projected points on the frames
    for i in range(num_points):
        if i == 0:
            continue
        cv2.circle(frame0, tuple(imagepoints_l[i].ravel().astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(frame1, tuple(imagepoints_m[i].ravel().astype(int)), 5, (0, 255, 0), -1)        
        cv2.circle(frame2, tuple(imagepoints_r[i].ravel().astype(int)), 5, (0, 255, 0), -1)

    # Resize frames to the same height (if necessary)
    height = min(frame0.shape[0], frame1.shape[0], frame2.shape[0])
    frame0 = cv2.resize(frame0, (int(frame0.shape[1] * height / frame0.shape[0]), height))
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))
    
    # Concatenate frames horizontally
    concatenated_frame = np.hstack((frame0, frame1, frame2))

    # Show the concatenated frame
    # cv2.imshow('Camera Stream', concatenated_frame)

    # c = cv2.waitKey(1)
    # print(f"c:{c}")
    # if c != -1:
    #     break
    
    time1 = time.time()
    
    # print(f"Time taken: {time1 - time0}")
    
    # change the 3d points by a random step
    step = np.random.randn(3) * 0.8
    test_3d_points += step

# Release the video capture objects and close the display window
cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()

print("------------------------------")
print(f"Average error 3 line: {error_3_total / errot_3_cnt}")
print(f"Average error 2 line: {error_2_total / error_2_cnt}")
print(f"Average error pf: {error_pf_total / error_pf_cnt}")
print(f"noise: {noise}")



