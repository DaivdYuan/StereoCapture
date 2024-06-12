import numpy as np
import cv2 as cv
import glob
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) - np.array([4, 2.5])
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('images/*.png')

KEYPOINT_NAMES = ['images/l.png', 'images/r.png', 'images/m.png']
keypoint_idx = {}
 
for fname in images:
    print(f"Processing {fname}")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        if fname in KEYPOINT_NAMES:
            keypoint_idx[fname] = len(objpoints)-1

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # try undistorting an image
# img = cv.imread('images/11.png')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# cv.imshow('img', img)
# cv.waitKey(0)

# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# x, y, w, h = roi
# dst_roi = dst[y:y+h, x:x+w]
# cv.imshow('img', dst_roi)
# cv.waitKey(0)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

    if i == 0:
    # print an example
        print("obj points: {}".format(objpoints[i][0]))
        print("image points: {}".format(imgpoints[i][0]))
        print("projected points: {}".format(imgpoints2[0]))
    
 
print( "total error: {}".format(mean_error/len(objpoints)) )

r_vec_l = rvecs[keypoint_idx['images/l.png']]
t_vec_l = tvecs[keypoint_idx['images/l.png']]

r_vec_r = rvecs[keypoint_idx['images/r.png']]
t_vec_r = tvecs[keypoint_idx['images/r.png']]

r_vec_m = rvecs[keypoint_idx['images/m.png']]
t_vec_m = tvecs[keypoint_idx['images/m.png']]

# save the calibration results
np.savez('calibration/calibration.npz', mtx=mtx, dist=dist, r_vec_l=r_vec_l, t_vec_l=t_vec_l, r_vec_r=r_vec_r, t_vec_r=t_vec_r, r_vec_m=r_vec_m, t_vec_m=t_vec_m)