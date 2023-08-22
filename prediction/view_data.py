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

def view_data(folder, config, stage='test', speed=10, fps=8):
    config, args = parse_config_args()
    dataset = WerpData(folder=folder, stage=stage, shuffle=False)
    plot_img = np.zeros((480, 640, 3))
    plotter = Plotter(frame=plot_img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.record_video:
        video = MovieWriter('videos/{}_{}.avi'.format(args.config, args.video_name), fps=fps)

    start_sec = 30
    end_sec = 90

    start_frame = start_sec * fps
    end_frame = end_sec * fps

    for i in range(1, len(dataset), speed):
        img, undistorted_img, ft, states, sensel, cam_params = dataset[i]

        img = img.to(device).unsqueeze(0)
        undistorted_img = undistorted_img.to(device).unsqueeze(0)
        sensel = sensel.to(device).unsqueeze(0)
        ft = ft.numpy()
        
        print('ft_gt: ', ft)

        img_viewable = undistorted_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_viewable = (img_viewable * 255).astype(np.uint8)
        gt_viewable = pressure_to_colormap(classes_to_scalar(sensel.squeeze(0).detach().cpu().numpy(), config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)

        gt_overlay = cv2.addWeighted(img_viewable, 1.0, gt_viewable, 1.0, 0.0)

        gt_overlay = cv2.resize(gt_overlay, (640, 480))
        cv2.putText(gt_overlay, 'ground truth', (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # concatenating the two images in the width direction
        cv2.imshow('gt_overlay', gt_overlay)

        # showing 3D plot
        plot_img = gt_overlay

        plot = plotter.visualize_ft(ft[0:3], ft[3:6], np.zeros(3), np.zeros(3), plot_img, collision_flag=False)
        cv2.imshow('plot', plot)

        if args.record_video:
                video.write_frame(gt_overlay)
        
        cv2.waitKey(1)

    if args.record_video:
        video.close()

if __name__ == '__main__':
    config, args = parse_config_args()
    view_data(folder=args.folder, config=config, stage=args.stage, speed=args.speed)