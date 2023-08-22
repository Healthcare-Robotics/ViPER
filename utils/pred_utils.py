import numpy as np
import torch
import torch.nn.functional as F
import os
from prediction.transforms import WerpTransforms
import cv2
from utils.sensel_utils import *
from utils.camera_utils import *
from utils.config_utils import *
from recording.aruco import ArucoGridEstimator

def preprocess_img(img, config, stage):
    # converts numpy image to a tensor for the network
    t = WerpTransforms(config.TRANSFORM, stage, config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms

    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.float()/255.0

    if transform is not None:
        img = transform(img)
    
    img = img.squeeze(0)
    return img

def preprocess_sensel(sensel, warped_img, config, stage='sensel'):
    # converts numpy sensel to a tensor for the network
    pose_estimator = ArucoGridEstimator(config)
    unwarped_img, homography, rvec, tvec, imgpts, ids = pose_estimator.get_cam_params_aruco(warped_img)

    sensel = convert_counts_to_kPa(sensel)
    sensel[sensel < 0] = 0
    
    if homography is not None:
        sensel = get_force_warped_to_img(unwarped_img, sensel, homography) # sensel_img is unwarped and resized
    else:
        sensel = np.zeros((config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y))

    sensel = torch.from_numpy(sensel)
    sensel = sensel.float()

    if config.FORCE_CLASSIFICATION:
        sensel = scalar_to_classes(sensel, config.FORCE_THRESHOLDS)

    t = WerpTransforms(config.TRANSFORM, stage, config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms

    if transform is not None:
        sensel = sensel.unsqueeze(0)
        sensel = transform(sensel)
        sensel = sensel.squeeze(0)

    return sensel

def predict(model, img, config):
    with torch.no_grad():
        pressure, ft = model(img.cuda())

        if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(pressure, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        else:
            force_pred_scalar = pressure.squeeze(1) * config.NORM_FORCE_REGRESS

    return force_pred_scalar.detach(), ft.detach()

def run_model(img, model, config):
    with torch.no_grad():
        if config.FORCE_CLASSIFICATION:
            force_pred_class = model(img.cuda())
            if isinstance(force_pred_class, tuple):
                force_pred_class = force_pred_class[0]

            force_pred_class = torch.argmax(force_pred_class, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        else:
            force_pred_scalar = model(img.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS

    return force_pred_scalar.detach()

def resnet_preprocessor(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if rgb.shape[2] == 12:
        mean = mean.repeat(4)
        std = std.repeat(4)

    rgb = rgb - mean
    rgb = rgb / std
    return rgb

def scalar_to_classes(scalar, thresholds):
    """
    Bins a float scalar into integer class indices. Could be faster, but is hopefully readable!
    :param scalar: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :return:
    """
    if torch.is_tensor(scalar):
        out = torch.zeros_like(scalar, dtype=torch.int64)
    else:
        out = np.zeros_like(scalar, dtype=np.int64)

    for idx, threshold in enumerate(thresholds):
        out[scalar >= threshold] = idx  # may overwrite the same value many times

    # if out.min() < 0:
    #     raise ValueError('Thresholds were not broad enough')

    return out

def classes_to_scalar(classes, thresholds):
    """
    Converts an integer class array into floating values. Obviously some discretization loss here
    :param classes: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :param final_value: if greater than the last threshold, fill in with this value
    :return:
    """
    if torch.is_tensor(classes):    # fill with negative ones
        out = -torch.ones_like(classes, dtype=torch.float)
    else:
        out = -np.ones_like(classes, dtype=np.float)

    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            val = thresholds[0]
        elif idx == len(thresholds) - 1:
            final_value = thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2    # Set it equal to the last value, plus half to gap to the previous thresh
            val = final_value
        else:
            val = (thresholds[idx] + thresholds[idx + 1]) / 2

        out[classes == idx] = val

    if out.min() < 0:
        raise ValueError('Thresholds were not broad enough')

    return out

def find_latest_checkpoint(config_name):
    """
    Finds the newest model checkpoint file, sorted by the index
    """
    all_folders = os.listdir('checkpoints/') # [config_x]

    # finding the folder that starts with the config name and has the highest index
    latest_folder_index = max([int(folder.split('_')[-1]) for folder in all_folders if folder.startswith(config_name)])

    all_models = os.listdir('checkpoints/{}_{}'.format(config_name, latest_folder_index))
    latest_model_index = max([int(model[:-4].split('_')[-1]) for model in all_models if model.startswith('model')])

    latest_file = 'checkpoints/{}_{}/model_{}.pth'.format(config_name, latest_folder_index, latest_model_index)
    print('Loading checkpoint file:', latest_file)

    return latest_file

def get_click_location(image):
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