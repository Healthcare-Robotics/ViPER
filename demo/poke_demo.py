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

HOME_POS_DICT = {'y': 0.1, 'roll': 0, 'pitch': 0, 'yaw': np.pi/2, 'gripper': 20}

class PokeDemo(LiveModel):
    def __init__(self):
        super().__init__(resolution=480)
        self.server.send_payload(HOME_POS_DICT)
        print('Moving to home')
        time.sleep(2)

        self.img_size = (448, 448)
        image = self.feed.get_frame()
        image = cv2.resize(image, self.img_size)

        self.prev_image = image

        bbox_size = 50
        self.tracker = cv2.TrackerCSRT_create()
        # self.tracker = cv2.TrackerKCF_create()

        self.y_target, self.x_target  = get_click_location(image)
        print('INITIAL X AND Y: ', self.x_target, self.y_target)
        self.x_target -= bbox_size // 2
        self.y_target -= bbox_size // 2
        print('CENTERED X AND Y: ', self.x_target, self.y_target)
        self.tracker.init(image, (self.x_target, self.y_target, bbox_size, bbox_size))

        # adding a rectangle centered at the click location
        cv2.rectangle(image, (self.x_target, self.y_target), (self.x_target + bbox_size, self.y_target + bbox_size), (0, 255, 0), 2)
        cv2.imshow('starting image', image)
        cv2.waitKey(0)
        self.y_target += Y_OFFSET
        self.state = 'aim_robot'

        # getting the average color in the bounding box
        self.avg_color = np.mean(image[self.y_target:self.y_target + bbox_size, self.x_target:self.x_target + bbox_size], axis=(0, 1))
        print('avg color', self.avg_color)

    def get_click_location(self, image):
        x_click = None
        y_click = None

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                nonlocal x_click, y_click
                x_click = x
                y_click = y

        cv2.imshow('image', image)
        cv2.setMouseCallback('image', click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f'Click y {y_click} x {x_click}')
        return y_click, x_click

    def find_maxima(self, force_image, threshold_abs=2):
        force_image = cv2.blur(force_image, (11, 11))
        coordinates = peak_local_max(force_image, min_distance=20, threshold_abs=threshold_abs, exclude_border=False)

        peak_list = []
        for i in range(coordinates.shape[0]):
            val = force_image[coordinates[i, 0], coordinates[i, 1]]
            peak_list.append({'pos': coordinates[i, :], 'val': val})

        return peak_list

    def closed_loop_to_target(self, force_image, target_yx, pos_dict, min_force, max_force):
        maxima = self.find_maxima(force_image)
        SLOP_X = 3
        SLOP_Y = 3

        x, y, z, gripper = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['gripper']
        sum_force = force_image.sum()

        if sum_force <= 0:
            self.server.send_payload({'z': z - 0.003})
        elif sum_force < min_force or len(maxima) < 2:
            self.server.send_payload({'z': z - 0.001})
        elif sum_force > max_force:
            self.server.send_payload({'z': z + 0.001})

            if len(maxima) < 2:
                return 9999

        mean_x = [pk['pos'][1] for pk in maxima]
        mean_y = [pk['pos'][2] for pk in maxima]

        mean_x = np.mean(mean_x)
        mean_y = np.mean(mean_y)

        if mean_x < target_yx[1] - SLOP_X:
            self.server.send_payload({'x': -0.005})
        elif mean_x > target_yx[1] + SLOP_X:
            self.server.send_payload({'x': 0.005})
        else:
            self.server.send_payload({'x': 0})

        if mean_y < target_yx[0] - SLOP_Y:
            self.server.send_payload({'y': pos_dict['y'] - 0.002})
        elif mean_y > target_yx[0] + SLOP_Y:
            self.server.send_payload({'y': pos_dict['y'] + 0.002})
        else:
            self.server.send_payload({'y': pos_dict['y']})

        err = np.linalg.norm([mean_y - target_yx[0], mean_x - target_yx[1]])

        print('target yx: ', target_yx)
        print('mean yx: ', mean_y, mean_x)

        return err

    def angle_robot(self, pos_dict):
        # x_center = self.img_size[1] // 2
        x_center = 240

        err = self.x_target - x_center

        eps = 2

        if err > eps:
            # self.server.send_payload({'theta': -0.025})
            self.server.send_payload({'y': pos_dict['y'] + 0.0025})
            time.sleep(0.25)
        elif err < -eps:
            # self.server.send_payload({'theta': 0.025})
            self.server.send_payload({'y': pos_dict['y'] - 0.0025})
            time.sleep(0.25)

        print('ERR: ', err)

        return err

    def aim_gripper(self, pos_dict, force_image, min_force, max_force, eps=5):
        maxima = self.find_maxima(force_image)

        x, y, z, gripper = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['gripper']
        sum_force = force_image.sum()

        # moving the robot back so we can reposition
        self.server.send_payload({'x': -0.02})
        time.sleep(1)

        if sum_force <= 0:
            self.server.send_payload({'x': +0.001})
        elif sum_force < min_force or len(maxima) < 2:
            self.server.send_payload({'x': +0.001})
        elif sum_force > max_force:
            self.server.send_payload({'x': -0.001})

            if len(maxima) < 2:
                return 9999

        print('maxima', maxima)

        mean_x = [pk['pos'][1] for pk in maxima]
        mean_y = [pk['pos'][2] for pk in maxima]

        mean_x = np.mean(mean_x)
        mean_y = np.mean(mean_y)

        err = [mean_y - self.y_target, mean_x - self.x_target]
        err_norm = np.linalg.norm(err)

        print('target: ', self.x_target, self.y_target)
        print('mean xy: ', mean_x, mean_y)
        print('CONTROLLER ERROR: ', err)

        # print('err x, y', mean_x - self.x_target, mean_y - self.y_target)

        k_p = 0.00005

        if len(maxima) >= 2: # only move horizontally if we have two peaks
            # if mean_x < self.x_target - eps:
            if err[1] < -eps:
                # self.server.send_payload({'y': pos_dict['y'] + 0.005})
                self.server.send_payload({'y': pos_dict['y'] + err[1] * k_p})
            # elif mean_x > self.x_target + eps:
            elif err[1] > eps:
                # self.server.send_payload({'y': pos_dict['y'] - 0.005})
                self.server.send_payload({'y': pos_dict['y'] + err[1] * k_p})

        # if mean_y < self.y_target - eps:
        if err[0] < -eps:
            # self.server.send_payload({'z': pos_dict['z'] - 0.002})
            self.server.send_payload({'z': pos_dict['z'] + err[0] * k_p})
        # elif mean_y > self.y_target + eps:
        elif err[0] > eps:
            # self.server.send_payload({'z': pos_dict['z'] + 0.002})
            self.server.send_payload({'z': pos_dict['z'] + err[0] * k_p})

        left_force_sum = force_image[:, :self.img_size[1] // 2].sum()
        right_force_sum = force_image[:, self.img_size[1] // 2:].sum()

        if left_force_sum > right_force_sum:
            self.server.send_payload({'theta': 0.005})
            self.server.send_payload({'yaw': pos_dict['yaw'] - 0.03})
        elif left_force_sum < right_force_sum:
            self.server.send_payload({'theta': -0.005})
            self.server.send_payload({'yaw': pos_dict['yaw'] + 0.03})
        
        return err

    def press(self,pos_dict):
        self.server.send_payload({'x': -0.05})
        time.sleep(1)
        self.server.send_payload({'gripper': -50})
        time.sleep(1)
        self.server.send_payload({'x': 0.1})
        time.sleep(2)

    def control_robot(self, force_pred):
        force_pred = force_pred.cpu().detach().numpy()
        enable_moving = True
 
        save_frames = []

        image = self.feed.get_frame()
        image = cv2.resize(image, (448, 448))

        sum_force = force_pred.sum()
        peak_list = self.find_maxima(force_pred)

        self.last_x_target, self.last_y_target = self.x_target, self.y_target

        ret, bbox = self.tracker.update(image)

        if ret:
            x, y, w, h = bbox
            # new x_target, y_target are the center of the bounding box
            self.x_target = int(x + w // 2)
            self.y_target = int(y + h // 2) + 5 # offset to account for camera perspective
            print('X AND Y TARGET: ', self.x_target, self.y_target)
        else:
            print('LOST TRACKING, USING OPTICAL FLOW')
            # self.x_target, self.y_target = self.last_x_target, self.last_y_target
            # if we lose tracking, we can use optical flow to estimate the new position
            # of the object

            lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            prev_img_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # p0 = np.array([[self.x_target, self.y_target]], dtype=np.float32)

            feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
            p0 = cv2.goodFeaturesToTrack(prev_img_gray, mask = None, **feature_params)

            # add x and y target to p0 to the beginning of the array
            p0 = np.insert(p0, 0, [[self.x_target, self.y_target]], axis=0)

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img_gray, image_gray, p0, None, **lk_params)
            print('P0: ', p0)
            print('P1: ', p1)
            self.x_target, self.y_target = p1[0][0][0], p1[0][0][1]
            print('OPTICAL FLOW: ', self.x_target, self.y_target)

        # cv2.circle(image, (x_target, y_target), 5, (0, 0, 255), -1)
        # cv2.imshow('current target', image)
        # cv2.waitKey(1)

        robot_ok, pos_dict = read_robot_status(self.client)

        if robot_ok and enable_moving:
            x, y, z, gripper = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['gripper']

            if self.state == 'aim_robot':
                ang_err = self.angle_robot(pos_dict)

                if abs(ang_err) < 20:
                    self.state = 'navigate'

            elif self.state == 'navigate':
                print('SUM_FORCE: ', sum_force)
                if sum_force > 1000 or (sum_force > 200 and len(peak_list) >= 2):
                    self.state = 'aim_gripper'
                else:
                    ang_err = self.angle_robot(pos_dict)
                    self.server.send_payload({'x':0.005})

            elif self.state == 'aim_gripper':
                # poke to get the position of the gripper
                # move back and make adjustments
                # repeat until the gripper is in the right position
                err = self.aim_gripper(pos_dict, force_pred, min_force=500, max_force=600, eps=5)
                # self.server.send_payload({'x':0.01})
                # time.sleep(1)

                if abs(err[0]) < 8 and abs(err[1]) < 8:
                    self.state = 'press'
                else:
                    self.state = 'navigate'

            elif self.state == 'press':
                print('pressing!')
                self.press(pos_dict)
                print('done pressing')
                exit()
                # self.state = 'rehome'

            elif self.state == 'rehome':
                self.server.send_payload(HOME_POS_DICT)
                time.sleep(2)
                enable_moving = False
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

        self.prev_image = image

        return self.x_target, self.y_target


if __name__ == "__main__":
    MIN_FORCE = 1000
    MAX_FORCE = 2000
    GRIPPER_CLOSE_TARGET = -20
    Y_OFFSET = 0
    STOP_CONTROL_Z_THRESH = 25

    pd = PokeDemo()
    pd.run_demo(control_func=pd.control_robot)