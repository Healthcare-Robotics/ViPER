import numpy as np
import cv2 as cv2
import glob
import os
from utils.config_utils import parse_config_args

def test_camera_calibration():
    config, args = parse_config_args()
        
    # paths
    calibration_folder = os.path.join('calibration', 'camera_data', config.CAMERA)
    coeff_path = os.path.join(calibration_folder, 'calibration_data.npz')

    # camera matrix and distortion coefficients
    mtx = np.load(coeff_path)['mtx']
    dist = np.load(coeff_path)['dist']

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        # cv2.imshow('img', img)
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imshow('dst', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    test_camera_calibration()