import time
import sys
import cv2

def read_robot_status(client):
    robot_status = client.receive_timeout(timeout=0.2)
    if robot_status is None:
        print('Robot not ok, dont have recent status packet')
        return False, None

    pos_dict = dict()
    pos_dict['lift_effort'] = robot_status['lift']['force']
    pos_dict['arm_effort'] = robot_status['arm']['force']
    pos_dict['roll_effort'] = robot_status['end_of_arm']['wrist_roll']['effort']
    pos_dict['pitch_effort'] = robot_status['end_of_arm']['wrist_pitch']['effort']
    pos_dict['yaw_effort'] = robot_status['end_of_arm']['wrist_yaw']['effort']
    pos_dict['gripper_effort'] = robot_status['end_of_arm']['stretch_gripper']['effort']

    # if pos_dict['lift_effort'] < -75:
    #     print('Robot not ok, too much lift force')
    #     return False, None

    pos_dict['x'] = robot_status['base']['x']
    pos_dict['theta'] = robot_status['base']['theta']
    pos_dict['z'] = robot_status['lift']['pos']
    pos_dict['y'] = robot_status['arm']['pos']
    pos_dict['roll'] = robot_status['end_of_arm']['wrist_roll']['pos']
    pos_dict['pitch'] = robot_status['end_of_arm']['wrist_pitch']['pos']
    pos_dict['yaw'] = robot_status['end_of_arm']['wrist_yaw']['pos']
    pos_dict['gripper'] = robot_status['end_of_arm']['stretch_gripper']['pos_pct']

    return True, pos_dict

def do_vertical_control_loop(server, z, sum_force, peak_list):
    print('sum_force: ', sum_force, 'peak_list: ', peak_list)
    if sum_force <= 0:
        server.send_payload({'z': z - 0.002})
    elif sum_force < 100 or len(peak_list) < 2:
        server.send_payload({'z': z - 0.001})
    elif sum_force > 3000:
        server.send_payload({'z': z + 0.001})

def keyboard_teleop(client, server, deltas, self=None): # enable_moving=True, stop=False):
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
        elif keycode == ord('1'):     # drive theta
            server.send_payload({'theta':deltas['theta']})
        elif keycode == ord('2'):     # drive theta
            server.send_payload({'theta':-deltas['theta']})

    # return enable_moving, stop

def level_robot(client, server, rpy_eps=0.1, grip_eps=0.5):
    server.send_payload({'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper':0})

    print("LEVELING ROBOT")
    time.sleep(1)

    robot_ok, pos_dict = read_robot_status(client)

    if not robot_ok:
        print("ROBOT NOT CONNECTED")
        return False
    
    elif abs(pos_dict['roll']) > rpy_eps or abs(pos_dict['pitch']) > rpy_eps or abs(pos_dict['yaw']) > rpy_eps or abs(pos_dict['gripper']) > grip_eps:
        print("ROBOT NOT LEVEL")
        return False

    else:
        print("ROBOT LEVELED")
        return True