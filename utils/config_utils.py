import argparse
import os
import yaml
import argparse
from types import SimpleNamespace

def load_config(config_name):
    config_path = os.path.join('./config', config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, default='default')
    parser.add_argument('--epoch', '-e', type=int, default=0, help='model epoch to load')
    parser.add_argument('--index', '-i', type=int, default=0, help='keeps track of training sessions using the same config')
    parser.add_argument('--fullscreen', '-fs', action='store_true', help='fullscreen live model figure')
    parser.add_argument('--folder', '-f', type=str, default=None, help='folder for data_capture or folder to pull data from if not live')
    parser.add_argument('--stage', '-s', type=str, default=None, help='train, test, or raw')
    parser.add_argument('--on_robot', '-r', action='store_true', help='run the model on the robot')
    parser.add_argument('--robot_state', action='store_true', help='record robot_state')
    parser.add_argument('--view', '-v', action='store_true', help='view camera and graphs')
    parser.add_argument('--view_3d', '-3d', action='store_true', help='view open3d output')
    parser.add_argument('--record_video', '-rec', action='store_true', help='record video')
    parser.add_argument('--video_name', '-vname', type=str, default='default', help='video name')
    parser.add_argument('--speed', '-sp', type=int, default=1, help='general speed multiplier')
    parser.add_argument('--xbox', '-x', action='store_true', help='use xbox controller')
    parser.add_argument('--sweep', '-swp', action='store_true', help='use wandb sweep')
    parser.add_argument('--skip_val', '-sv', action='store_true', help='only validate on last epoch')
    parser.add_argument('--use_latest', '-ul', action='store_true', help='use the latest checkpoint for a given config')
    parser.add_argument('--saliency', '-sal', action='store_true', help='generate saliency map')
    parser.add_argument('--use_ft', '-ft', action='store_true', help='use force/torque sensor')
    parser.add_argument('--use_sensel', '-sl', action='store_true', help='use sensel')

    args = parser.parse_args()
    return load_config(args.config), args