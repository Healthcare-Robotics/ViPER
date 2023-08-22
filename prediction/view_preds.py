import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import sys
from utils.config_utils import *
from utils.camera_utils import *
from utils.sensel_utils import *
from utils.pred_utils import *
from recording.view_pressure import get_force_overlay_img
import json
from pathlib import Path
from prediction.loader import WerpData
from prediction.plotter import Plotter
from paper.blur_saliency import gen_blur_saliency

def view_preds(folder, stage='test', speed=10, fps=5):
    config, args = parse_config_args()
    dataset = WerpData(folder=folder, stage=stage, shuffle=False)
    plot_img = np.zeros((480, 640, 3))
    plotter = Plotter(frame=plot_img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_latest:
        model_path = find_latest_checkpoint(config.CONFIG_NAME)
    else:
        model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    if args.record_video:
        video = MovieWriter('videos/{}_{}.avi'.format(args.config, args.video_name), fps=fps)

    start_sec = 30
    end_sec = 90

    start_frame = start_sec * fps
    end_frame = end_sec * fps

    for i in range(1, len(dataset), args.speed):
        img, undistorted_img, ft, states, sensel, cam_params = dataset[i]

        img = img.to(device).unsqueeze(0)
        undistorted_img = undistorted_img.to(device).unsqueeze(0)
        sensel = sensel.to(device).unsqueeze(0)
        ft = ft.numpy()
        
        pressure_pred, ft_pred = model(undistorted_img)
        ft_pred = ft_pred['bottleneck_logits'].squeeze(0).detach().cpu().numpy()
        print('ft_pred: ', ft_pred)
        print('ft_gt: ', ft)
        print('force gt magnitude: ', np.linalg.norm(ft[0:3]))

        if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(pressure_pred, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        else:
            force_pred_scalar = pressure_pred.squeeze(1) * config.NORM_FORCE_REGRESS

        output_viewable = pressure_to_colormap(force_pred_scalar.squeeze(0).detach().cpu().numpy(), colormap=cv2.COLORMAP_INFERNO)
        img_viewable = undistorted_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_viewable = (img_viewable * 255).astype(np.uint8)
        gt_viewable = pressure_to_colormap(classes_to_scalar(sensel.squeeze(0).detach().cpu().numpy(), config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)

        pred_overlay = cv2.addWeighted(img_viewable, 1.0, output_viewable, 1.0, 0.0)
        gt_overlay = cv2.addWeighted(img_viewable, 1.0, gt_viewable, 1.0, 0.0)

        pred_overlay = cv2.resize(pred_overlay, (640, 480))
        #cv2.putText(pred_overlay, 'estimated', (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        gt_overlay = cv2.resize(gt_overlay, (640, 480))
        #cv2.putText(gt_overlay, 'ground truth', (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # concatenating the two images in the width direction
        output_img = np.concatenate((pred_overlay, gt_overlay), axis=1) # (480, 1280, 3)
        cv2.imshow('output', output_img)

        # frame_folder = '~/werp/werp/paper/grasp_fig_frames/'
        # cv2.imwrite('paper/grasp_fig_frames/frame_{}.png'.format(i), pred_overlay)
        # print('saving to: ', os.path.join(frame_folder, 'frame_{}.png'.format(i)))

        # showing 3D plot
        # overlay_plot = cv2.resize(output_img, (640, 240))
        # vertically centering the plot
        # plot_img[120:360, :, :] = overlay_plot

        plot = plotter.visualize_ft(ft[0:3], ft[3:6], ft_pred[0:3], ft_pred[3:6], plot_img, collision_flag=False)
        # plot = plotter.visualize_ft(ft[0:3], ft[3:6], np.zeros((3,)), np.zeros((3,)), plot_img, collision_flag=False)
        plot[100:580, 100:1380, :] = output_img
        
        # bottom right
        plot[-449:-1, -449:-1, :] = gt_viewable

        # bottom left
        plot[-449-400:-1-400, 1:449, :] = img_viewable

        cv2.imshow('plot', plot)

        # cv2.imshow('sensel', sensel_viewable)
        # cv2.waitKey(0)
        # cv2.imshow('img_viewable', img_viewable)
        # cv2.waitKey(0)
        # cv2.imshow('gt_overlay', cv2.addWeighted(img_viewable, 1.0, sensel_viewable, 1.0, 0.0))
        # cv2.waitKey(1)

        if args.saliency:
            saliency_map = gen_blur_saliency(model, undistorted_img)
            cv2.imshow('saliency', saliency_map)

        if args.record_video:
            if args.saliency:
                video.write_frame(saliency_map)
            else:
                video.write_frame(pred_overlay)
        
        cv2.waitKey(1)

    if args.record_video:
        video.close()

if __name__ == '__main__':
    config, args = parse_config_args()
    view_preds(folder=args.folder, stage=args.stage)
    