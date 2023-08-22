import numpy as np
import cv2 as cv2
import glob
import os
import argparse
from utils.config_utils import parse_config_args


class CameraCalibration:
    def __init__(self):
        self.config, self.args = parse_config_args()
        
        # paths
        self.calibration_folder = os.path.join('calibration', 'camera_data', self.config.CAMERA)
        self.calib_result_path = os.path.join(self.calibration_folder, 'calibresult.png')
        self.coeff_path = os.path.join(self.calibration_folder, 'calibration_data.npz')

        if not os.path.exists(self.calibration_folder):
            os.makedirs(self.calibration_folder)

        if not os.path.exists(os.path.join(self.calibration_folder, 'calibration_images')):
            os.makedirs(os.path.join(self.calibration_folder, 'calibration_images'))

        self.image_folder = os.path.join(self.calibration_folder, 'calibration_images')

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.deleted_frames = []

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*9,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    def collect_calib_images(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # setting i to the name with the highest number in calibration_images
        if len(os.listdir(self.image_folder)) > 0:
            i = np.max([int(x[:-4].split('_')[1]) for x in os.listdir(self.image_folder)]) + 1
        else:
            i = 0
            print('starting at image: ', i)

        # streaming video
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()
            cv2.imshow('img', img)
            cv2.imwrite(os.path.join(self.image_folder, 'img_'+str(i)+'.png'), img)
            if cv2.waitKey(333) & 0xFF == ord('q'):
                break

            i += 1

        cap.release()

    def evaluate_calib_images(self):
        # for frame in sorted(os.listdir(self.image_folder)):
        for frame in sorted(glob.glob(os.path.join(self.image_folder, '*.png'))):
            print('IMAGE PATH: ', frame)
            img = cv2.imread(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners2, ret)
                cv2.imshow('img', img)
                if cv2.waitKey(0) & 0xFF == ord(' '):
                    # add frame to deleted_frames
                    self.deleted_frames.append(frame)
                    continue # skip bad frames
            else:
                # add frame to deleted_frames
                self.deleted_frames.append(frame)
                continue

        cv2.destroyAllWindows()

        # delete bad frames
        for frame in self.deleted_frames:
            print('deleting frame: ', frame)
            os.remove(frame)

    def unwarp_image(self):
        self.collect_calib_images()
        self.evaluate_calib_images()

        # reading the last image to undistort
        img = cv2.imread(sorted(glob.glob(os.path.join(self.image_folder, '*.png')))[-1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(self.calib_result_path, dst)

        # save the camera matrix and distortion coefficients
        np.savez(self.coeff_path, mtx=mtx, dist=dist)

        # checking the error
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "total error: {}".format(mean_error/len(self.objpoints)) )

if __name__ == '__main__':
    calib = CameraCalibration()
    calib.unwarp_image()