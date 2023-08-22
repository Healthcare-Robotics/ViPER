import cv2
import time
import os
import numpy as np

def get_camera_calibration(config):
        # paths
        calibration_folder = os.path.join('calibration', 'camera_data', config.CAMERA)
        coeff_path = os.path.join(calibration_folder, 'calibration_data.npz')

        # camera matrix and distortion coefficients
        mtx = np.load(coeff_path)['mtx']
        dist = np.load(coeff_path)['dist']

        return mtx, dist

def set_camera_parameters(config, cam_obj):
        # setting exposure
        cam_obj.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cam_obj.set(cv2.CAP_PROP_EXPOSURE, 100)

        # # setting window size
        # cam_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cam_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)