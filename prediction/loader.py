import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import os
import cv2
import sys
from prediction.transforms import WerpTransforms
from utils.config_utils import *
from utils.sensel_utils import *
from utils.pred_utils import *
from utils.camera_utils import *
import json
from pathlib import Path

class WerpData(data.Dataset):
    def __init__(self, folder='/data/raw/', stage='raw', shuffle=True):
        self.config, self.args = parse_config_args()
        self.stage = stage

        # to handle old configs
        if type(folder) == str:
            self.root = [os.path.join(os.getcwd(), folder)]
        
        elif type(folder) == list:
            self.root = folder

        print('Loading data from', self.root)
        
        self.dataset = self.get_data(shuffle=shuffle)

    def __getitem__(self, index):
        # obtaining file paths
        img_path, ft_path, state_path, sensel_path, cam_param_path = self.dataset[index]

        # loading ft data and subtracting offset
        ft = np.load(ft_path, allow_pickle=True)
        offset_folder = Path(ft_path).parent.parent
        ft_offset = np.load(os.path.join(offset_folder, 'ft_calibration.npy'))
        ft = ft - ft_offset
        ft = torch.from_numpy(ft)

        # loading and formatting image
        img = cv2.imread(img_path)

        # loading sensel data
        sensel = np.load(sensel_path)
        sensel = preprocess_sensel(sensel, img, self.config, stage='sensel')

        # undistorting image and resizing for the network. sensel_img is the image that the sensel data is warped to
        mtx, dist = get_camera_calibration(self.config)
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

        img = preprocess_img(img, self.config, self.stage)
        undistorted_img = preprocess_img(undistorted_img, self.config, self.stage)
               
        # loading robot state data
        states = torch.tensor([])

        # setting robot states to 0 if not found
        with open(state_path, 'r') as f:
            robot_state = json.load(f)
            if robot_state is None:
                robot_state = {'gripper': 0.0, 'z': 0.0, 'y': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'lift_effort': 0.0, 'arm_effort': 0.0, 'roll_effort': 0.0, 'pitch_effort': 0.0, 'yaw_effort': 0.0, 'gripper_effort': 0.0}
            for key in robot_state.keys():
                if type(robot_state[key]) != float or not np.isfinite(robot_state[key]) or np.isnan(robot_state[key]) or np.abs(robot_state[key]) > 100:
                    # print('weird val: ', key, robot_state[key])
                    robot_state[key] =  0.0

        states = torch.tensor([])

        for state in self.config.ROBOT_STATES:
            if state in robot_state:
                states = torch.cat((states, torch.tensor([robot_state[state]])), dim=0)
            else:
                states = torch.cat((states, torch.tensor([0.0])), dim=0)

        # loading camera parameters
        cam_params = dict()
        homography = np.load(cam_param_path, allow_pickle=True)['homography']
        cam_params['homography'] = torch.from_numpy(homography)

        rvec = np.load(cam_param_path, allow_pickle=True)['rvec']
        cam_params['rvec'] = torch.from_numpy(rvec)

        tvec = np.load(cam_param_path, allow_pickle=True)['tvec']
        cam_params['tvec'] = torch.from_numpy(tvec)

        imgpts = np.load(cam_param_path, allow_pickle=True)['imgpts']
        cam_params['imgpts'] = torch.from_numpy(imgpts)

        ids = np.load(cam_param_path, allow_pickle=True)['ids']
        cam_params['ids'] = torch.tensor([ids.shape[0]])

        if 'flip' in self.config.TRANSFORM and np.random.rand() > 0.5:
            img = img.flip(2)
            undistorted_img = undistorted_img.flip(2)
            sensel = sensel.flip(1)
            # mirroring ft data horizontally
            ft[1] = -ft[1] # Fy = horizontal force
            ft[3] = -ft[3] # Tx = yaw torque
            ft[5] = -ft[5] # Tz = roll torque

        # sensel_viewable = pressure_to_colormap(classes_to_scalar(sensel.numpy(), self.config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)
        # img_viewable = undistorted_img.permute(1, 2, 0).numpy()
        # img_viewable = (img_viewable * 255).astype(np.uint8)

        # cv2.imshow('sensel', sensel_viewable)
        # cv2.imshow('img_viewable', img_viewable)
        # cv2.imshow('gt_overlay', cv2.addWeighted(img_viewable, 1.0, sensel_viewable, 1.0, 0.0))
        # cv2.waitKey(0)
       
        # print('img: ', img.shape)
        # print('undistorted_img: ', undistorted_img.shape)
        # print('ft: ', ft.shape)
        # print('states: ', states.shape)
        # print('sensel: ', sensel.shape)

        # img: distorted img (3, 448, 448)
        # undistorted_img: undistorted img (3, 448, 448)
        # ft: (6,)
        # states: (len(config.ROBOT_STATES),)
        # sensel: sensel projected into undistorted image space: (448, 448)
        # cam_params: dict of camera parameters (homography, rvec, tvec, imgpts, ids)
        return img, undistorted_img, ft, states, sensel, cam_params

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        ft_names = []
        state_names = []
        sensel_names = []
        cam_param_names = []
        self.dataset = []

        # crawling the directory to sort the data modalities 
        for folder in self.root:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    # saving (time, path)
                    parent = root.split('/')[-1]
                    if parent == 'cam' and file.endswith('.jpg'):
                        img_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'ft' and file.endswith('.npy'):
                        ft_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'state' and file.endswith('.txt'):
                        state_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'sensel' and file.endswith('.npy'):
                        sensel_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'cam_params' and file.endswith('.npz'):
                        cam_param_names.append((float(file[:-4]), os.path.join(root, file)))                        

        for name in [img_names, ft_names, state_names, sensel_names, cam_param_names]:
            name.sort(key=lambda x: x[0])

        self.timestamps = [x[0] for x in img_names]

        list_sizes = [len(img_names), len(ft_names), len(state_names), len(sensel_names), len(cam_param_names)]
        print('Dataset sizes before filtering: ', list_sizes)

        # checking if number of elements in each list is the same
        if len(set(list_sizes)) == 1:
            for i in range(len(img_names)):
                # # filtering out data where number of elements in cam_params['ids'] < 3
                params = np.load(cam_param_names[i][1])
                num_ids = params['ids'].shape[0]
                # print('num_ids: ', num_ids)
                if self.stage in ['train', 'val', 'test']: # TODO: remove filter for videos
                    if num_ids > 3:
                    #     # print('ids: ', params['ids'])
                        self.dataset.append((img_names[i][1], ft_names[i][1], state_names[i][1], sensel_names[i][1], cam_param_names[i][1]))
                elif self.stage in ['weak_train', 'weak_val', 'weak_test']:
                    self.dataset.append((img_names[i][1], ft_names[i][1], state_names[i][1], sensel_names[i][1], cam_param_names[i][1]))
                else:
                    print('Error: Invalid stage.')
                    sys.exit(1)
        else:
            print('Error: Number of images, ft data, and state data do not match. Check folder sizes.')
            sys.exit(1)

        print('Dataset sizes after filtering: ', len(self.dataset))

        if shuffle:
            np.random.shuffle(self.dataset)
        else:
            self.dataset = np.array(self.dataset)

        # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...]
        return self.dataset

if __name__ == '__main__':
    config, args = parse_config_args()
    dataset = WerpData(folder=config.TEST_FOLDER, stage='test')
    print("frames in folder: ", len(dataset.dataset))
    for i in range(100):
        img, undistorted_img, ft, states, sensel, cam_params = dataset[i]
        print('ft: ', ft)
        print('states: ', states)
        print('sensel: ', sensel.shape)
        print('homography: ', cam_params['homography'])
        print('rvec: ', cam_params['rvec'])
        print('tvec: ', cam_params['tvec'])
        print('imgpts: ', cam_params['imgpts'])
        print('ids: ', cam_params['ids'])

        img = img.permute(1, 2, 0)
        cv2.imshow('img', np.array(img))
        cv2.waitKey(0)