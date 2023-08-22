import cv2.aruco as aruco
import os
import pickle
import cv2
import numpy as np
from utils.config_utils import parse_config_args
from utils.camera_utils import *
from robot.robot_utils import *
import time
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client

class ArucoGridEstimator:
    def __init__(self, config, draw=False):
        self.cameraMatrix , self.distCoeffs = get_camera_calibration(config)
        self.draw = draw

        if self.cameraMatrix is None or self.distCoeffs is None:
            raise IOError('Calibration issue. Remove ./calibration.pckl and recalibrate your camera with generate_camera_calibration.py.')

        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()  # Constant parameters used in Aruco methods
        self.ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 2
        self.ARUCO_PARAMETERS.perspectiveRemovePixelPerCell = 12
        self.ARUCO_PARAMETERS.minMarkerPerimeterRate = 0.01

        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)

        self.sensel_corners_2D = np.float32([[0, 105], [185, 105], [185, 0], [0, 0]])

        if hasattr(config, 'ARUCO'):
            self.aruco_config = config.ARUCO
        else:
            self.aruco_config = 'default'
          
        self.current_cam_param_time = time.time()

        self.sensel_w = 0.235
        self.sensel_h = 0.135

        if self.aruco_config == 'default':
            self.sensel_origin_x = -0.035
            self.sensel_origin_y = 0.157
            self.sensel_z = 0.00375
            self.phi = 0

            self.board = aruco.CharucoBoard_create(
                squaresX=5,
                squaresY=3,
                squareLength=0.033295,
                markerLength=0.02535,
                dictionary=self.ARUCO_DICT)
                
        elif self.aruco_config == 'angled_6x2':
            self.sensel_origin_x = 0.00
            self.sensel_origin_y = 0.062
            self.sensel_z = -0.06425
            self.phi = -45 

            self.board = aruco.CharucoBoard_create(
            squaresX=6,
            squaresY=2,
            squareLength=0.04,
            markerLength=0.0308,
            dictionary=self.ARUCO_DICT)

        elif self.aruco_config == 'angled_6x2_90':
            self.sensel_origin_x = 0.005
            self.sensel_origin_y = 0.01
            self.sensel_z = -0.14325
            self.phi = -90 

            self.board = aruco.CharucoBoard_create(
            squaresX=6,
            squaresY=2,
            squareLength=0.04,
            markerLength=0.0308,
            dictionary=self.ARUCO_DICT)

    def get_cam_params_aruco(self, img):
        # takes in a distorted image and returns the camera parameters and an undistorted image
        output_img = np.array(img)
        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)   # grayscale image

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)  # Detect Aruco markers
        # print('num good IDs, num rejected IDs', len(corners), len(rejectedImgPoints))
        
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image=gray,
                board=self.board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejectedImgPoints,
                cameraMatrix=self.cameraMatrix,
                distCoeffs=self.distCoeffs)

        if self.draw:
            # Outline all of the markers detected in our image
            output_img = aruco.drawDetectedMarkers(output_img, corners, borderColor=(0, 0, 255))

        homography = None
        rvec = None
        tvec = None
        imgpts = None

        if ids is not None: # and len(ids) > 3:
            # print('Got N IDs from aruco estimation:', len(ids))
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
            pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.board, self.cameraMatrix, self.distCoeffs, rvec=None, tvec=None)

            if pose:
                self.sensel_corners_3D = np.float32([[self.sensel_origin_x + self.sensel_w, self.sensel_origin_y + self.sensel_h, self.sensel_z],
                                                    [self.sensel_origin_x, self.sensel_origin_y + self.sensel_h, self.sensel_z],
                                                    [self.sensel_origin_x, self.sensel_origin_y, self.sensel_z],
                                                    [self.sensel_origin_x + self.sensel_w, self.sensel_origin_y, self.sensel_z]])

                # rotation matrix that rotates by phi about the x-axis
                rotation_matrix_x = np.array([[1, 0, 0],
                                              [0, np.cos(self.phi * np.pi / 180), -np.sin(self.phi * np.pi / 180)],
                                              [0, np.sin(self.phi * np.pi / 180), np.cos(self.phi * np.pi / 180)]]
                                              )
                
                # rotating sensel corners
                self.sensel_corners_3D = np.matmul(self.sensel_corners_3D, rotation_matrix_x)

                # undistort the image
                output_img = cv2.undistort(output_img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
                # projecting 3D points onto the undistorted image
                imgpts, jac = cv2.projectPoints(self.sensel_corners_3D, rvec, tvec, self.cameraMatrix, np.zeros((4, 1)))
                homography, _ = cv2.findHomography(self.sensel_corners_2D, imgpts[:, 0, :])

                for c_idx in range(4):
                    start_point = tuple(imgpts[c_idx, 0, :].astype(int))
                    end_point = tuple(imgpts[(c_idx + 1) % 4, 0, :].astype(int))
                    
                    if self.draw:
                        output_img = cv2.line(output_img, start_point, end_point, (0, 0, 255), 4)
            else:
                output_img = cv2.undistort(output_img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
        else:
            output_img = cv2.undistort(output_img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
            
        self.current_cam_param_time = time.time()

        return output_img, homography, rvec, tvec, imgpts, ids


if __name__ == "__main__":
    config, args = parse_config_args()

    pose_estimator = ArucoGridEstimator(config, draw=True)

    client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
    server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)
    time.sleep(0.5)

    cap = cv2.VideoCapture(0)
    set_camera_parameters(config, cap)

    while True:
        ret, img = cap.read()
        if img is not None:
            img, homography, rvec, tvec, imgpts, ids = pose_estimator.get_cam_params_aruco(img)
            cv2.imshow('QueryImage', img)

        keyboard_teleop(client, server, config.ACTION_DELTA_DICT)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        # # changing the origin of the sensel
        # if key == ord('x'):
        #     pose_estimator.sensel_origin_x += 0.001
        # if key == ord('1'):
        #     pose_estimator.sensel_origin_x -= 0.001
        # if key == ord('y'):
        #     pose_estimator.sensel_origin_y += 0.001
        # if key == ord('2'):
        #     pose_estimator.sensel_origin_y -= 0.001
        # if key == ord('z'):
        #     pose_estimator.sensel_z += 0.001
        # if key == ord('3'):
        #     pose_estimator.sensel_z -= 0.001
        # if key == ord('p'):
        #     pose_estimator.phi += 1
        # if key == ord('4'):
        #     pose_estimator.phi -= 1
        
        # print('sensel_origin_x', pose_estimator.sensel_origin_x)
        # print('sensel_origin_y', pose_estimator.sensel_origin_y)
        # print('sensel_z', pose_estimator.sensel_z)
        # print('phi', pose_estimator.phi)

    cv2.destroyAllWindows()
    cap.release()