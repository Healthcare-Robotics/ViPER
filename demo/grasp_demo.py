import cv2
import numpy as np
import torch
from recording.camera import Camera
from recording.ft import FTCapture
import recording.sensel_wrapper as sensel_wrapper
from robot.robot_utils import *
from utils.config_utils import *
from utils.camera_utils import *
from utils.ft_utils import *
from utils.pred_utils import *
from utils.sensel_utils import *
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from recording.aruco import ArucoGridEstimator
from prediction.live_model import LiveModel
from skimage.feature import peak_local_max

HOME_POS_DICT = {'y': 0.1, 'z': 0.9, 'roll': 0, 'pitch': -0.5, 'yaw': 0, 'gripper': 25}

class GraspDemo(LiveModel):
    def __init__(self):
        super().__init__()
        self.server.send_payload(HOME_POS_DICT)
        print('Moving to home')
        time.sleep(2)
        image = self.feed.get_frame()
        image = cv2.resize(image, (448, 448))

        bbox_size = 24
        self.tracker = cv2.TrackerCSRT_create()
        y_target, x_target  = get_click_location(image)
        x_target -= bbox_size // 2
        y_target -= bbox_size // 2
        self.tracker.init(image, (x_target, y_target, bbox_size, bbox_size))

        # adding a rectangle centered at the click location
        cv2.rectangle(image, (x_target, y_target), (x_target + bbox_size, y_target + bbox_size), (0, 255, 0), 2)
        cv2.imshow('starting image', image)
        cv2.waitKey(0)
        y_target += Y_OFFSET
        self.state = 'hover'

    def find_maxima(self, force_image, threshold_abs=2):
        # gets a list of maxima locations and magnitudes
        force_image = cv2.blur(force_image, (11, 11))
        coordinates = peak_local_max(force_image, min_distance=20, threshold_abs=threshold_abs, exclude_border=False)

        peak_list = []
        for i in range(coordinates.shape[0]):
            val = force_image[coordinates[i, 0], coordinates[i, 1]]
            peak_list.append({'pos': coordinates[i, :], 'val': val})

        return peak_list # peak_list is a list of dicts, each dict has a 'pos' and 'val' key

    def closed_loop_to_target(self, server, force_image, target_yx, pos_dict, min_force, max_force, skip_vertical=False):
        # moves the robot one step in the direction of the target
        # also regulates the vertical position of the robot wrt the pressure

        maxima = self.find_maxima(force_image)
        SLOP_X = 3
        SLOP_Y = 3

        x, y, z, gripper = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['gripper']
        sum_force = force_image.sum()

        if not skip_vertical:
            if sum_force <= 0:
                server.send_payload({'z': z - 0.003})
            elif sum_force < min_force or len(maxima) < 2:
                server.send_payload({'z': z - 0.001})
            elif sum_force > max_force:
                server.send_payload({'z': z + 0.001})

            if len(maxima) < 2:
                return 9999

        print('maxima', maxima)

        mean_x = [pk['pos'][1] for pk in maxima]
        mean_y = [pk['pos'][2] for pk in maxima]

        mean_x = np.mean(mean_x)
        mean_y = np.mean(mean_y)

        err = [mean_x - target_yx[1], mean_y - target_yx[0]]

        if mean_x < target_yx[1] - SLOP_X:
            server.send_payload({'x': -0.005})
        elif mean_x > target_yx[1] + SLOP_X:
            server.send_payload({'x': 0.005})
        else:
            server.send_payload({'x': 0})

        if mean_y < target_yx[0] - SLOP_Y:
            server.send_payload({'y': pos_dict['y'] - 0.003})
        elif mean_y > target_yx[0] + SLOP_Y:
            server.send_payload({'y': pos_dict['y'] + 0.003})
        else:
            server.send_payload({'y': pos_dict['y']})

        err = np.linalg.norm([mean_y - target_yx[0], mean_x - target_yx[1]])

        print('target yx: ', target_yx)
        print('mean yx: ', mean_y, mean_x)

        return err

    def control_robot(self, force_pred):
        force_pred = force_pred.cpu().detach().numpy()
        enable_moving = True

        save_frames = []

        robot_ok, pos_dict = read_robot_status(self.client)

        image = self.feed.get_frame()
        image = cv2.resize(image, (448, 448))

        sum_force = force_pred.sum()
        peak_list = self.find_maxima(force_pred)

        ret, bbox = self.tracker.update(image)
        x, y, w, h = bbox

        # new x_target, y_target are the center of the bounding box
        x_target = int(x + w // 2)
        y_target = int(y + h // 2) - 5 # offset to account for camera perspective

        # cv2.circle(image, (x_target, y_target), 5, (0, 0, 255), -1)
        # cv2.imshow('current target', image)
        # cv2.waitKey(1)

        if robot_ok and enable_moving:
            x, y, z, gripper = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['gripper']

            self.server.send_payload({'pitch': HOME_POS_DICT['pitch']})
        
            if self.state == 'hover':
                do_vertical_control_loop(self.server, z, sum_force, peak_list, min_force=100, max_force=3000)

                if sum_force > 100 and len(peak_list) >= 2:
                    self.state = 'approach'
            elif self.state == 'approach':
                err = self.closed_loop_to_target(self.server, force_pred, (y_target, x_target), pos_dict, min_force=MIN_FORCE, max_force=MAX_FORCE) # min_force=5000, max_force=7000)

                if err < 5:
                    self.state = 'grasp'
            elif self.state == 'grasp':
                min_vertical_grasp = 4000
                max_vertical_grasp = 6000

                if (sum_force > min_vertical_grasp and sum_force < max_vertical_grasp) or pos_dict['gripper'] < 0:
                    self.server.send_payload({'gripper': gripper - 2})
                else:
                    do_vertical_control_loop(self.server, z, sum_force, peak_list, min_force=min_vertical_grasp, max_force=max_vertical_grasp)
                # err = self.closed_loop_to_target(self.server, force_pred, (y_target, x_target), pos_dict, min_force=min_vertical_grasp, max_force=max_vertical_grasp, skip_vertical=gripper < STOP_CONTROL_Z_THRESH)

                if gripper < GRIPPER_CLOSE_TARGET:
                    self.state = 'lift'
            elif self.state == 'lift':
                target_z = HOME_POS_DICT['z']
                self.server.send_payload({'z': target_z})
                time.sleep(2)
                self.server.send_payload({'gripper':HOME_POS_DICT['gripper']})
                if z > target_z - 0.01:
                    self.state = 'rehome'
            elif self.state == 'rehome':
                self.server.send_payload(HOME_POS_DICT)
                time.sleep(2)
                exit()

        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord(' '):     # toggle moving
            enable_moving = not enable_moving
        if keycode == ord('h'):     # go home
            enable_moving = False
            self.server.send_payload(HOME_POS_DICT)
        if keycode == ord(','):     # drive X
            self.server.send_payload({'x': -0.020})
        if keycode == ord('.'):     # drive X
            self.server.send_payload({'x': 0.020})
        if keycode == ord('q'):
            exit()

        return x_target, y_target


if __name__ == "__main__":
    MIN_FORCE = 1000
    MAX_FORCE = 2000
    GRIPPER_CLOSE_TARGET = -35
    Y_OFFSET = 0
    STOP_CONTROL_Z_THRESH = 25

    gd = GraspDemo()
    gd.run_demo(control_func=gd.control_robot)