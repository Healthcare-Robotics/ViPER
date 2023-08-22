from multiprocessing import synchronize
import re
import cv2
import sys
import numpy as np
import time
from datetime import datetime
import os
from recording.camera import Camera
from recording.ft import FTCapture
import recording.sensel_wrapper as sensel_wrapper
from recording.aruco import ArucoGridEstimator
import argparse
import threading
# import keyboard
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from utils.config_utils import *
from utils.camera_utils import *
from utils.ft_utils import *
from utils.sensel_utils import *
from utils.pred_utils import *
# from prediction.plotter import Plotter
import json

class DataCapture:
    def __init__(self):
        self.config, self.args = parse_config_args()
        resource = 3 if self.args.on_robot else 0
        self.feed = Camera(resolution=480, resource=resource, view=self.args.view)
        self.ft = FTCapture()
        self.sensel = sensel_wrapper.SenselWrapper()
        self.data_folder = 'data'

        self.cameraMatrix , self.distCoeffs = get_camera_calibration(self.config)

        self.pose_estimator = ArucoGridEstimator(self.config, draw=True)

        self.delta_lin = 0.02
        self.delta_ang = 0.1
        self.enable_moving = True
        self.stop = False

        if self.args.robot_state and not self.args.xbox:
            self.client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
            self.server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)
            time.sleep(0.5)
            # leveling robot
            success = level_robot(self.client, self.server)
            if not success:
                sys.exit()

        # making directories for data if they doesn't exist
        for folder in ['data', 'data/raw', 'data/train', 'data/test', 'data/weak_train', 'data/weak_test']:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # counting the number of folders in the stage folder beginning with args.folder
        folders = os.listdir(os.path.join(self.data_folder, self.args.stage))
        
        if len(folders) == 0:
            folder_count = 0
        else:
            # folder_count = len([f for f in folders if re.match(self.args.folder, f)])
            folder_count = len([f for f in folders if f.split('/')[-1].startswith(self.args.folder)])

        self.args.folder = self.args.folder + '_' + str(folder_count)

        # setting folder names as class attributes (self.<name>_folder)
        for name in ['cam', 'ft', 'state', 'sensel', 'cam_params']:
            folder = os.path.join(self.data_folder, self.args.stage, self.args.folder, name)
            setattr(self, name + '_folder', folder)
        
        # making directories for data if they doesn't exist
        for folder in [self.cam_folder, self.ft_folder, self.state_folder, self.sensel_folder, self.cam_params_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # saving ft calibration
        self.ft_offset = self.ft.get_ft()

        if np.abs(self.ft_offset.mean()) < 1e-3:
            print('FT NOT CONNECTED')
            sys.exit()

        print('CALIBRATING FT: ', self.ft_offset)
        time.sleep(0.5)

        np.save(os.path.join(self.data_folder, self.args.stage, self.args.folder, 'ft_calibration.npy'), self.ft_offset)

    def capture_data(self):
        # get data snapshots
        frame = self.feed.get_frame()
        ft_data = self.ft.get_ft()
        sensel_data = self.sensel.cur_force_array

        print('ft: ', ft_data - self.ft_offset)

        if self.args.robot_state and not self.args.xbox:
            robot_ok, self.pos_dict = read_robot_status(self.client)
        else:
            self.pos_dict = None

        self.disp_img, homography, rvec, tvec, imgpts, ids = self.pose_estimator.get_cam_params_aruco(frame)

        # set file names based on timestamps
        image_name = '{:.3f}'.format(self.feed.current_frame_time).replace('.', '_') + '.jpg'
        ft_name = '{:.3f}'.format(self.ft.current_frame_time).replace('.', '_')
        state_name = '{:.3f}'.format(self.ft.current_frame_time).replace('.', '_') + '.txt'
        sensel_name = '{:.3f}'.format(self.sensel.cur_force_timestamp).replace('.', '_') + '.npy'
        cam_param_name = '{:.3f}'.format(self.pose_estimator.current_cam_param_time).replace('.', '_') # + '.npy'

        # save data to machine
        if self.args.stage in ['train', 'test', 'weak_train', 'weak_test', 'raw']:
            cv2.imwrite(os.path.join(self.cam_folder, image_name), frame)
            np.save(os.path.join(self.ft_folder, ft_name), ft_data)
            np.save(os.path.join(self.sensel_folder, sensel_name), sensel_data)

            with open(os.path.join(self.state_folder, state_name), 'w') as file:
                if self.args.robot_state:
                    file.write(json.dumps(self.pos_dict))
                else:
                    file.write(json.dumps({"no_robot_state":None}))

            if homography is not None:
                np.savez(os.path.join(self.cam_params_folder, cam_param_name), homography=homography, rvec=rvec, tvec=tvec, imgpts=imgpts, ids=ids)
            else:
                # np.savez(os.path.join(self.cam_params_folder, cam_param_name), homography=np.array([]), rvec=np.array([]), tvec=np.array([]), imgpts=np.array([]), ids=np.array([]))
                np.savez(os.path.join(self.cam_params_folder, cam_param_name), homography=-np.ones((3,3)), rvec=-np.ones((3,1)), tvec=-np.ones((3,1)), imgpts=-np.ones((4,1,2)), ids=-np.ones((4,1)))

        else:
            print('Invalid stage argument. Please choose train, test, weak_train, weak_test, or raw')
            sys.exit(1)

        result = {
                  'frame':frame,
                  'frame_time':self.feed.current_frame_time,
                  'ft_frame':ft_data,
                  'ft_frame_time':self.ft.current_frame_time,
                  'robot_state':self.pos_dict,
                  'sensel_frame':sensel_data,
                  'sensel_frame_time':self.sensel.cur_force_timestamp,
                  'cam_params':{
                              'homography':homography,
                              'rvec':rvec,
                              'tvec':tvec,
                              'imgpts':imgpts,
                              'ids':ids
                              },
                  'cam_param_time':self.pose_estimator.current_cam_param_time
                  }

        if self.feed.view:
            disp_time = str(round(self.feed.current_frame_time - self.feed.first_frame_time, 3))
            sensel_viewable = pressure_to_colormap(self.sensel.cur_force_array)
          
            # making the sensel the same size as the camera image
            sensel_viewable = cv2.resize(sensel_viewable, (self.disp_img.shape[1], self.disp_img.shape[0]))
            cv2.putText(self.disp_img, disp_time, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            # concat the two images in the width direction
            combined = np.concatenate((self.disp_img, sensel_viewable), axis=1)
            cv2.imshow("combined", combined)

            # gt = np.array(ft_data)
            # force_gt = gt[0:3]
            # torque_gt = gt[3:6]
            # fig = self.plotter.visualize_ft(force_gt=force_gt, torque_gt=torque_gt, force_pred=np.zeros((3,)), torque_pred=np.zeros((3,)), frame=frame, collision_flag=False, view_3D=False)
            # cv2.imshow('live view', fig)

        if self.args.robot_state and not self.args.xbox:
            # need to access robot state to control with keyboard
            keyboard_teleop(self.client, self.server, self.config.ACTION_DELTA_DICT, self)

        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop = True

        print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

        return result

if __name__ == "__main__":
    cap = DataCapture()
    delay = []

    while not cap.stop:
        data = cap.capture_data()
        delay.append(data['ft_frame_time'] - data['frame_time'])

    folder_sizes = [len(files) for r, d, files in os.walk(os.path.join(cap.data_folder, cap.args.stage, cap.args.folder))][1:]
    folder_names = [r.split('/')[-1] for r, d, files in os.walk(os.path.join(cap.data_folder, cap.args.stage, cap.args.folder))][1:]
    folder_dict = dict(zip(folder_names, folder_sizes))
    
    print('folder sizes: ', folder_dict)

    if len(set(folder_sizes)) > 1:
        print('ERROR: not all folders have the same number of files')
        print('missing files in: ', [k for k, v in folder_dict.items() if v != max(folder_sizes)])
        
    print('saved results to {}'.format(os.path.join(cap.data_folder, cap.args.stage, cap.args.folder)))
    print("delay avg:", np.mean(delay))
    print("delay std:", np.std(delay))
    print("delay max:", np.max(delay))