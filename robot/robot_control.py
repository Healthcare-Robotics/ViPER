#!/usr/bin/env python
import time
import math
import stretch_body.robot
import zmq_client
import zmq_server
import argparse
import keyboard
from robot_utils import *

class StretchManipulation:
    def __init__(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()

        self.arm_vel = 0.15
        self.arm_accel = 0.15

        self.wrist_vel = 0.000000001 
        self.wrist_accel = 0.0000000001

        self.client = zmq_client.SocketClient(zmq_client.IP_DESKTOP, port=zmq_client.PORT_COMMAND_SERVER)

        self.server = zmq_server.SocketServer(port=zmq_client.PORT_STATUS_SERVER)
        self.thread = zmq_client.DaemonStoppableThread(0.02, target=self.publish_status_loop, name='status_sender')
        self.thread.start()
        print('Socket threads started')

        parser = argparse.ArgumentParser()
        parser.add_argument('--local', action='store_true', help='Use keyboard on robot')

        self.args = parser.parse_args()

        # self.level_robot()

        ACTION_DELTA_DICT = {'x': 0.02, 'y': 0.02, 'z': 0.005, 'roll': 0.1, 'pitch': 0.1, 'yaw': 0.05, 'gripper': 5, 'theta': 0.1}

        while True:
            try:
                delta_dict = self.client.receive_blocking()
                print('Received data', delta_dict)
                self.navigate_robot_abs(delta_dict)
                if self.args.local:
                    self.keyboard_teleop(self.client, self.server, ACTION_DELTA_DICT)

            except KeyboardInterrupt:
                self.robot.stop()

    def publish_status_loop(self):
        status = self.robot.get_status()
        self.server.send_payload(status)

    def navigate_robot_abs(self, input_dict):
        print('Lift force', self.robot.lift.status['force'])

        if 'x' in input_dict:
            print
            self.robot.base.translate_by(input_dict['x'], self.arm_vel, self.arm_accel)
        if 'y' in input_dict:
            self.robot.arm.move_to(input_dict['y'], self.arm_vel, self.arm_accel)
        if 'z' in input_dict:
            self.robot.lift.move_to(input_dict['z'], self.arm_vel, self.arm_accel)

        if 'roll' in input_dict:
            self.robot.end_of_arm.move_to('wrist_roll', input_dict['roll'], self.wrist_vel, self.wrist_accel)
        if 'pitch' in input_dict:
            self.robot.end_of_arm.move_to('wrist_pitch', input_dict['pitch'], self.wrist_vel, self.wrist_accel)
        if 'yaw' in input_dict:
            self.robot.end_of_arm.move_to('wrist_yaw', input_dict['yaw'], self.wrist_vel, self.wrist_accel)
        if 'gripper' in input_dict:
            print('moving gripper to ', input_dict['gripper'])
            self.robot.end_of_arm.move_to('stretch_gripper', input_dict['gripper'], self.wrist_vel, self.wrist_accel)

        if 'theta' in input_dict:
            print('rotating by ', input_dict['theta'])
            self.robot.base.rotate_by(input_dict['theta'])

        self.robot.push_command()

    def level_robot(self, input_dict={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper':0}):
        if 'x' in input_dict:
            self.robot.base.translate_by(input_dict['x'], self.arm_vel, self.arm_accel)
        if 'y' in input_dict:
            self.robot.arm.move_to(input_dict['y'], self.arm_vel, self.arm_accel)
        if 'z' in input_dict:
            self.robot.lift.move_to(input_dict['z'], self.arm_vel, self.arm_accel)

        if 'roll' in input_dict:
            self.robot.end_of_arm.move_to('wrist_roll', input_dict['roll'], self.wrist_vel, self.wrist_accel)
        if 'pitch' in input_dict:
            self.robot.end_of_arm.move_to('wrist_pitch', input_dict['pitch'], self.wrist_vel, self.wrist_accel)
        if 'yaw' in input_dict:
            self.robot.end_of_arm.move_to('wrist_yaw', input_dict['yaw'], self.wrist_vel, self.wrist_accel)
        if 'gripper' in input_dict:
            print('moving gripper to ', input_dict['gripper'])
            self.robot.end_of_arm.move_to('stretch_gripper', input_dict['gripper'], self.wrist_vel, self.wrist_accel)

        self.robot.push_command()
        time.sleep(1)

    def keyboard_teleop(self, client, server, deltas): # enable_moving=True, stop=False):
        robot_ok, pos_dict = read_robot_status(client)
        
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord('q') and hasattr(self, 'stop'):     # stop
                self.stop = True
        if keycode == ord(' ') and hasattr(self, 'enable_moving'):     # toggle moving
            self.enable_moving = not self.enable_moving

        move_ok = pos_dict is not None and (self is None or (hasattr(self, 'enable_moving') and self.enable_moving))

        if move_ok:
            if keycode == ord(']'):     # drive X
                server.send_payload({'x':-deltas['x']})
            elif keycode == ord('['):     # drive X
                server.send_payload({'x':deltas['x']})
            elif keycode == ord('a'):     # drive Y
                server.send_payload({'y':pos_dict['y'] - deltas['y']})
            elif keycode == ord('d'):     # drive Y
                server.send_payload({'y':pos_dict['y'] + deltas['y']})
            elif keycode == ord('s'):     # drive Z
                server.send_payload({'z':pos_dict['z'] - deltas['z']})
            elif keycode == ord('w'):     # drive Z
                server.send_payload({'z':pos_dict['z'] + deltas['z']})
            elif keycode == ord('u'):     # drive roll
                server.send_payload({'roll':pos_dict['roll'] - deltas['roll']})
            elif keycode == ord('o'):     # drive roll
                server.send_payload({'roll':pos_dict['roll'] + deltas['roll']})
            elif keycode == ord('k'):     # drive pitch
                server.send_payload({'pitch':pos_dict['pitch'] - deltas['pitch']})
            elif keycode == ord('i'):     # drive pitch
                server.send_payload({'pitch':pos_dict['pitch'] + deltas['pitch']})
            elif keycode == ord('l'):     # drive yaw
                server.send_payload({'yaw':pos_dict['yaw'] - deltas['yaw']})
            elif keycode == ord('j'):     # drive yaw
                server.send_payload({'yaw':pos_dict['yaw'] + deltas['yaw']})
            elif keycode == ord('b'):     # drive gripper
                server.send_payload({'gripper':pos_dict['gripper'] - deltas['gripper']})
            elif keycode == ord('n'):     # drive gripper
                server.send_payload({'gripper':pos_dict['gripper'] + deltas['gripper']})

if __name__ == '__main__':
    sm = StretchManipulation()