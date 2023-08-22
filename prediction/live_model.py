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
from prediction.plotter import Plotter
from recording.capture_data import DataCapture

class LiveModel:
    def __init__(self, resolution=480):
        self.config, self.args = parse_config_args()
        resource = 3 if self.args.on_robot else 0

        self.stage = 'test'

        self.cap = DataCapture()
        self.feed = self.cap.feed
        self.ft = self.cap.ft
        self.sensel = self.cap.sensel
        self.pose_estimator = self.cap.pose_estimator
        self.client = self.cap.client
        self.server = self.cap.server

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(self.args.config, self.args.index, self.args.epoch))
        self.model = torch.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.enable_moving = True
        self.stop = False

        self.plot_img = np.zeros((480, 640, 3))
        self.plotter = Plotter(frame=self.plot_img)

        if self.args.record_video:
            self.video = MovieWriter('videos/{}_{}.avi'.format(self.args.config, self.args.video_name), fps=8)

         
    def sleep_and_record(self, duration):
        robot_ok, pos_dict = read_robot_status(self.client)
        start = time.time()
        while time.time() - start < duration:
            if self.args.record_video:
                delay_start = time.time()
                data = self.cap.capture_data()

                delay = time.time() - delay_start
                if (1 / self.video_fps - delay) != 0:
                    time.sleep(abs(1 / self.video_fps - delay)) # regulating loop time to self.video_fps frame rate
            else:
                time.sleep(0.001)

    def run_demo(self, control_func):
        while True:
            robot_ok, pos_dict = read_robot_status(self.client)
            data = self.cap.capture_data()

            # frame = self.feed.get_frame()
            frame = data['frame']
            mtx, dist = get_camera_calibration(self.config)
            undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

            # ft_data = self.ft.get_ft()
            ft_data = data['ft_frame']
            # sensel_data = self.sensel.cur_force_array
            sensel_data = data['sensel_frame']
            sensel_input = preprocess_sensel(sensel_data, frame, self.config).unsqueeze(0)

            img_input = preprocess_img(frame, self.config, self.stage).unsqueeze(0)
            undistorted_img_input = preprocess_img(undistorted_frame, self.config, self.stage).unsqueeze(0)

            img_input = img_input.to(self.device)
            undistorted_img_input = undistorted_img_input.to(self.device)
            sensel_input = sensel_input.to(self.device)

            pressure_pred, ft_pred = self.model(undistorted_img_input)
            ft_pred = ft_pred['bottleneck_logits'].squeeze(0).detach().cpu().numpy()
            # ft_pred -= self.ft_offset
            ft_data -= self.cap.ft_offset
            print('ft_pred: ', ft_pred)
            print('ft_gt: ', ft_data)
            print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

            if self.config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(pressure_pred, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, self.config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = self.pressure_pred.squeeze(1) * self.config.NORM_FORCE_REGRESS

            output_viewable = pressure_to_colormap(force_pred_scalar.squeeze(0).detach().cpu().numpy(), colormap=cv2.COLORMAP_INFERNO)
            img_viewable = undistorted_img_input.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_viewable = (img_viewable * 255).astype(np.uint8)
            gt_viewable = pressure_to_colormap(classes_to_scalar(sensel_input.squeeze(0).detach().cpu().numpy(), self.config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)

            pred_overlay = cv2.addWeighted(img_viewable, 1.0, output_viewable, 1.0, 0.0)
            gt_overlay = cv2.addWeighted(img_viewable, 1.0, gt_viewable, 1.0, 0.0)

            cv2.putText(gt_overlay, 'ground truth', (280, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            x_target, y_target = control_func(force_pred_scalar)
            cv2.circle(pred_overlay, (int(x_target), int(y_target)), 5, (0, 0, 255), -1)

            cv2.imshow('output', pred_overlay)
            cv2.waitKey(1)

            if self.args.record_video:
                self.video.write_frame(pred_overlay)

            self.video_fps = self.feed.frame_count / (time.time() - self.feed.first_frame_time)
             
            force_gt = ft_data[0:3]
            torque_gt = ft_data[3:6]
            force_pred = ft_pred[0:3]
            torque_pred = ft_pred[3:6]

            if np.linalg.norm(force_gt) > 30:
                print('force too high, don\'t hurt robot')
                break

            error = ft_pred - ft_data
            # self.pred_hist = np.concatenate(([ft_pred], self.pred_hist), axis=0)
            # self.gt_hist = np.concatenate(([ft_data], self.gt_hist), axis=0)
            # self.timestamp_hist = np.concatenate(([self.ft.current_frame_time], self.timestamp_hist), axis=0)

            # if self.args.view:
            #     self.fig = self.plotter.visualize_ft(force_gt, torque_gt, force_pred, torque_pred, self.frame, self.collision_flag)
            #     # self.fig = self.plotter.visualize_ft(force_gt_robot_frame, torque_gt_robot_frame, force_pred_robot_frame, torque_pred_robot_frame, self.frame, self.collision_flag)

            #     cv2.imshow('figure', self.fig)
            #     self.keyboard_teleop()

            #     if self.args.record_video:
            #         self.result.write(self.fig)

            if self.stop:
                break

        # # removing initialized values from history
        # self.pred_hist = self.pred_hist[:-1]
        # self.gt_hist = self.gt_hist[:-1]


    def run_model(self):
        while not self.stop:

            data = self.cap.capture_data()

            frame = data['frame']
            mtx, dist = get_camera_calibration(self.config)
            undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

            ft_data = data['ft_frame']
            sensel_data = data['sensel_frame']
            sensel_input = preprocess_sensel(sensel_data, frame, self.config).unsqueeze(0)

            img_input = preprocess_img(frame, self.config, self.stage).unsqueeze(0)
            undistorted_img_input = preprocess_img(undistorted_frame, self.config, self.stage).unsqueeze(0)

            img_input = img_input.to(self.device)
            undistorted_img_input = undistorted_img_input.to(self.device)
            sensel_input = sensel_input.to(self.device)

            pressure_pred, ft_pred = self.model(undistorted_img_input)
            ft_pred = ft_pred['bottleneck_logits'].squeeze(0).detach().cpu().numpy()
            ft_data -= self.cap.ft_offset
            print('ft_pred: ', ft_pred)
            print('ft_gt: ', ft_data)
            any_contact_gt = np.linalg.norm(ft_data[0:3]) > 3
            print('any contact gt: ', any_contact_gt)

            print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

            if self.config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(pressure_pred, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, self.config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = self.pressure_pred.squeeze(1) * self.config.NORM_FORCE_REGRESS

            output_viewable = pressure_to_colormap(force_pred_scalar.squeeze(0).detach().cpu().numpy(), colormap=cv2.COLORMAP_INFERNO)
            img_viewable = undistorted_img_input.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_viewable = (img_viewable * 255).astype(np.uint8)
            gt_viewable = pressure_to_colormap(classes_to_scalar(sensel_input.squeeze(0).detach().cpu().numpy(), self.config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)

            pred_overlay = cv2.addWeighted(img_viewable, 1.0, output_viewable, 1.0, 0.0)
            gt_overlay = cv2.addWeighted(img_viewable, 1.0, gt_viewable, 1.0, 0.0)

            # pred_overlay = cv2.undistort(pred_overlay, mtx, dist, None, mtx)
            # gt_overlay = cv2.undistort(gt_overlay, mtx, dist, None, mtx)

            pred_overlay = cv2.resize(pred_overlay, (640, 480))
            cv2.putText(pred_overlay, 'pred', (280, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            gt_overlay = cv2.resize(gt_overlay, (640, 480))
            cv2.putText(gt_overlay, 'ground truth', (280, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # concatenate the two images in the width direction
            output_img = np.concatenate((pred_overlay, gt_overlay), axis=1)
            # cv2.imshow('output', output_img)

            # showing 3D plot
            plot = self.plotter.visualize_ft(ft_data[0:3], ft_data[3:6], ft_pred[0:3], ft_pred[3:6], self.plot_img, collision_flag=False)
            plot[100:580, 100:1380, :] = output_img
            cv2.imshow('plot', plot)

            if self.args.record_video:
                self.video.write_frame(output_img)

            keyboard_teleop(self.client, self.server, self.config.ACTION_DELTA_DICT, self)

        if self.args.record_video:
            self.video.close()

if __name__ == '__main__':
    live_model = LiveModel()
    live_model.run_model()