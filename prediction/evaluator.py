import os.path

import torch
import numpy as np
from prediction.loader import *
import prediction.trainer
from tqdm import tqdm
# import recording.util as util
from utils.pred_utils import *
from prediction.model_builder import build_model
import torchmetrics
import pprint
import datetime
# from recording.util import json_write
import pandas as pd
import torch.nn.functional as F
# from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.ops import box_area, box_convert, box_iou
import random
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import cv2
import utils.recording_utils as util

class InvididualStats:
    def __init__(self, seq_reader=None):
        self.seq_reader = seq_reader
        self.cols = ['participant', 'action', 'camera_idx', 'key', 'val']
        # self.df = pd.DataFrame(columns=self.cols)

        self.data = list()

    def update(self, preds, target, participants, actions, camera_idxs, any_contact_pred, any_contact_gt):
        assert preds.shape == target.shape

        high = torch.maximum(preds, target).sum((1, 2))
        low = torch.minimum(preds, target).sum((1, 2))
        gt_sum = target.sum((1, 2))
        pred_sum = preds.sum((1, 2))

        high = high.detach().cpu().numpy()
        low = low.detach().cpu().numpy()
        gt_sum = gt_sum.detach().cpu().numpy()
        pred_sum = pred_sum.detach().cpu().numpy()

        preds_numpy = preds.detach().cpu().numpy()
        gt_numpy = target.detach().cpu().numpy()

        any_contact_pred = any_contact_pred.detach().cpu().numpy()
        any_contact_gt = any_contact_gt.detach().cpu().numpy()

        for i in range(len(participants)):
            self.add_to_dict('numerator', low[i], participants[i], actions[i], camera_idxs[i])
            self.add_to_dict('denominator', high[i], participants[i], actions[i], camera_idxs[i])
            self.add_to_dict('gt_sum', gt_sum[i], participants[i], actions[i], camera_idxs[i])
            self.add_to_dict('pred_sum', pred_sum[i], participants[i], actions[i], camera_idxs[i])
            self.add_to_dict('gt_any_contact', int(any_contact_gt[i]), participants[i], actions[i], camera_idxs[i])
            self.add_to_dict('pred_any_contact', int(any_contact_pred[i]), participants[i], actions[i], camera_idxs[i])

            if self.seq_reader is not None:
                pred_force = self.seq_reader.get_force_cropped_pressure_img(camera_idxs[i], config, preds_numpy[i, :, :])
                gt_force = self.seq_reader.get_force_cropped_pressure_img(camera_idxs[i], config, gt_numpy[i, :, :])
                self.add_to_dict('gt_force', gt_force, participants[i], actions[i], camera_idxs[i])
                self.add_to_dict('pred_force', pred_force, participants[i], actions[i], camera_idxs[i])

    def add_to_dict(self, key, value, participant, action, camera_idx):
        point = (participant, action, camera_idx, key, value)
        self.data.append(point)

    def save_dict(self, config, network_name=''):
        df = pd.DataFrame(self.data, columns=self.cols)
        d = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_filename = os.path.join('data', 'eval', f"{os.path.basename(network_name)}_individual_{d}.csv")
        df.to_csv(out_filename)


class VolumetricIOU(torchmetrics.Metric):
    """
    This calculates the IoU summed over the entire dataset, then averaged. This means an image with no
    GT or pred force will contribute none to this metric.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape

        # # visualizing the first in the batch
        # cv2.imshow('IOU preds', preds[0].detach().cpu().numpy().astype(np.float32))
        # # cv2.imshow('IOU preds*255', preds[0].detach().cpu().numpy().astype(np.float32) * 255)
        # cv2.imshow('IOU target', target[0].detach().cpu().numpy().astype(np.float32))
        # cv2.waitKey(0)

        high = torch.maximum(preds, target)
        low = torch.minimum(preds, target)

        self.numerator += torch.sum(low)
        self.denominator += torch.sum(high)

    def compute(self):
        # print(f"Volumetric IOU: {self.numerator} / {self.denominator} = {self.numerator / self.denominator}")
        return self.numerator / self.denominator


class ContactIOU(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape
        assert preds.dtype == torch.long    # Make sure we're getting ints

        bool_pred = preds > 0
        bool_gt = target > 0

        self.numerator += torch.sum(bool_gt & bool_pred)
        self.denominator += torch.sum(bool_gt | bool_pred)

    def compute(self):
        return self.numerator / self.denominator

class RootMeanSquaredError(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_squares", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.sum_squares += torch.sum((preds - target) ** 2)
        self.count += target.numel()

    def compute(self):
        return torch.sqrt(self.sum_squares / self.count)

def reset_metrics(all_metrics):
    for key, metric in all_metrics.items():
        metric.reset()


def print_metrics(all_metrics, config, network_name='', save=True):
    out_dict = dict()
    for key, metric in all_metrics.items():
        val = metric.compute().item()
        print(key, val)
        out_dict[key] = val

    if save:
        d = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_filename = os.path.join('data', 'eval', f"{os.path.basename(network_name)}_{d}.txt")
        # json_write(out_filename, out_dict, auto_mkdir=True)


CONTACT_THRESH = 1.0


def run_metrics(all_metrics, pressure_gt, pressure_pred, ft, ft_pred, config):
    # Takes CUDA BATCHES as input
    pressure_pred = pressure_pred.detach()  # just in case

    contact_pred = (pressure_pred > CONTACT_THRESH).long()
    contact_gt = (pressure_gt > CONTACT_THRESH).long()

    # all_metrics['contact_iou'](contact_pred, contact_gt)
    all_metrics['contact_iou_old'](contact_pred, contact_gt)
    # all_metrics['contact_f1'](contact_pred, contact_gt)

    all_metrics['mse'](pressure_pred, pressure_gt)
    all_metrics['mae'](pressure_pred, pressure_gt)
    all_metrics['vol_iou'](pressure_pred, pressure_gt)

    any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
    any_contact_gt = torch.sum(contact_gt, dim=(1, 2)) > 0

    all_metrics['contact_accuracy'](any_contact_pred, any_contact_gt)

    force_pred = ft_pred['bottleneck_logits'][:, :3]
    force_gt = ft[:, :3]
    torque_pred = ft_pred['bottleneck_logits'][:, 3:]
    torque_gt = ft[:, 3:]

    any_ft_contact_gt_1N = torch.norm(force_gt, dim=1) > 1
    any_ft_contact_gt_2N = torch.norm(force_gt, dim=1) > 2
    any_ft_contact_gt_3N = torch.norm(force_gt, dim=1) > 3

    # all_metrics['contact_accuracy'](any_contact_pred, any_ft_contact_gt_3N)
    all_metrics['temporal_accuracy_1N'](any_contact_pred, any_ft_contact_gt_1N)
    all_metrics['temporal_accuracy_2N'](any_contact_pred, any_ft_contact_gt_2N)
    all_metrics['temporal_accuracy_3N'](any_contact_pred, any_ft_contact_gt_3N)
    
    # we only want to compute temporal accuracy on the frames where the ground truth force is not within the deadband
    # ignore index is 2

    # any_contact_gt_deadband_1_3 is false when < 1N, true when > 3N, and 2 when in the deadband
    any_contact_gt_deadband_1_3 = 2 * torch.ones_like(any_ft_contact_gt_1N)
    any_contact_gt_deadband_1_3[torch.norm(force_gt, dim=1) < 1] = 0
    any_contact_gt_deadband_1_3[torch.norm(force_gt, dim=1) > 3] = 1

    # any_contact_gt_deadband_2_3 is false when < 2N, true when > 3N, and 2 when in the deadband
    any_contact_gt_deadband_2_3 = 2 * torch.ones_like(any_ft_contact_gt_2N)
    any_contact_gt_deadband_2_3[torch.norm(force_gt, dim=1) < 2] = 0
    any_contact_gt_deadband_2_3[torch.norm(force_gt, dim=1) > 3] = 1

    all_metrics['temporal_accuracy_deadband_1_3'](any_contact_pred, any_contact_gt_deadband_1_3)
    all_metrics['temporal_accuracy_deadband_2_3'](any_contact_pred, any_contact_gt_deadband_2_3)

    log_pred = torch.log1p(pressure_pred)
    log_gt = torch.log1p(pressure_gt)

    all_metrics['log_vol_iou'](log_pred, log_gt)

    force_pred = ft_pred['bottleneck_logits'][:, :3]
    force_gt = ft[:, :3]
    torque_pred = ft_pred['bottleneck_logits'][:, 3:]
    torque_gt = ft[:, 3:]

    all_metrics['force_rmse'](force_pred, force_gt)
    all_metrics['torque_rmse'](torque_pred, torque_gt)

def setup_metrics(device):
    all_metrics = dict()

    all_metrics['contact_iou'] = ContactIOU().to(device)
    all_metrics['contact_iou_old'] = torchmetrics.JaccardIndex(2).to(device)
    all_metrics['contact_accuracy'] = torchmetrics.Accuracy().to(device)
    all_metrics['mse'] = torchmetrics.MeanSquaredError().to(device)
    all_metrics['mae'] = torchmetrics.MeanAbsoluteError().to(device)
    all_metrics['vol_iou'] = VolumetricIOU().to(device)
    all_metrics['temporal_accuracy_1N'] = torchmetrics.Accuracy().to(device)
    all_metrics['temporal_accuracy_2N'] = torchmetrics.Accuracy().to(device)
    all_metrics['temporal_accuracy_3N'] = torchmetrics.Accuracy().to(device)
    all_metrics['temporal_accuracy_deadband_1_3'] = torchmetrics.Accuracy(ignore_index=2, num_classes=3).to(device)
    all_metrics['temporal_accuracy_deadband_2_3'] = torchmetrics.Accuracy(ignore_index=2, num_classes=3).to(device)
    # all_metrics['contact_f1'] = torchmetrics.F1Score(num_classes=2, average='macro', mdmc_average='samplewise').to(device)
    # all_metrics['contact_f1_1N'] = torchmetrics.F1Score(num_classes=2, average='macro', mdmc_average='samplewise').to(device)
    # all_metrics['contact_f1_2N'] = torchmetrics.F1Score(num_classes=2, average='macro', mdmc_average='samplewise').to(device)
    # all_metrics['contact_f1_3N'] = torchmetrics.F1Score(num_classes=2, average='macro', mdmc_average='samplewise').to(device)
    all_metrics['log_vol_iou'] = VolumetricIOU().to(device)
    all_metrics['force_rmse'] = RootMeanSquaredError().to(device)
    all_metrics['torque_rmse'] = RootMeanSquaredError().to(device)
    return all_metrics

def evaluate_labeled(config, force_zero_guesser=False):
    model_dict = build_model(config, device, ['val'])
    val_sampler = RandomSampler(model_dict['val_dataset'], replacement=True, num_samples=1000)
    val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, sampler=val_sampler)
    # model = model_dict['model']
    # model.eval()
    criterion = model_dict['criterion']
    val_metrics = setup_metrics(device)

    loss_meter = util.AverageMeter('Loss', ':.4e')
    reset_metrics(val_metrics)

    model_name = '{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch)
    checkpoint_path = os.path.join(config.MODEL_DIR, model_name)
    best_model = torch.load(checkpoint_path)
    best_model.eval()

    with torch.no_grad():
        # for idx, data in enumerate(tqdm(val_loader)):
        for img, undistorted_img, ft, states, sensel, cam_params in tqdm(val_loader):

            # # view img and sensel
            # sensel_viewable = pressure_to_colormap(classes_to_scalar(sensel[0].numpy().astype(np.uint8), config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)
            # img_viewable = undistorted_img[0].permute(1, 2, 0).numpy()
            # img_viewable = (img_viewable * 255).astype(np.uint8)
            # cv2.imshow('gt_overlay_val', cv2.addWeighted(img_viewable, 1.0, sensel_viewable, 1.0, 0.0))
            # cv2.waitKey(0)

            image_gpu = undistorted_img.cuda()
            force_gt_gpu = sensel.cuda()
            batch_size = undistorted_img.shape[0]
            ft = ft.cuda()

            force_estimated, ft_pred = best_model(image_gpu, alpha=0)

            # #############
            # # ZERO GUESSER!!!!!!!!!!!!!!!!!!!!
            # force_estimated *= 0.0
            # # setting prediction to first class
            # force_estimated[:, 0, :, :] = 1.0
            # ft_pred['bottleneck_logits'] *= 0.0
            # #############

            loss = criterion(force_estimated, force_gt_gpu)

            loss_meter.update(loss.item(), batch_size)

            if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(force_estimated, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = force_estimated.squeeze(1) * config.NORM_FORCE_REGRESS

            # force_gt_scalar = force_gt_gpu
            force_gt_scalar = sensel.cuda()

            run_metrics(val_metrics, force_gt_scalar, force_pred_scalar, ft, ft_pred, config)

    print_metrics(val_metrics, config, checkpoint_path)

def evaluate_weak(config, force_test_on_test=False, force_zero_guesser=False):
    model_dict = build_model(config, device, ['val_weak'])
    val_sampler = RandomSampler(model_dict['val_weak_dataset'], replacement=True, num_samples=1000)
    val_weak_loader = DataLoader(model_dict['val_weak_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, sampler=val_sampler)
    weak_val_metrics = setup_metrics(device)

    reset_metrics(weak_val_metrics)

    model_name = '{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch)
    checkpoint_path = os.path.join(config.MODEL_DIR, model_name)
    best_model = torch.load(checkpoint_path)
    best_model.eval()

    with torch.no_grad():
        for img, undistorted_img, ft, states, sensel, cam_params in tqdm(val_weak_loader):
            undistorted_img = undistorted_img.cuda()
            weak_labels_source_gt = ft.cuda()
            ft = ft.cuda()

            force_estimated, ft_pred = best_model(undistorted_img, alpha=0)

            # #############
            # ZERO GUESSER!!!!!!!!!!!!!!!!!!!!
            # force_estimated *= 0.0
            # # setting prediction to first class
            # force_estimated[:, 0, :, :] = 1.0
            # ft_pred['bottleneck_logits'] *= 0.0
            # #############

            if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(force_estimated, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

            else:
                force_pred_scalar = force_estimated.squeeze(1) * config.NORM_FORCE_REGRESS

            force_gt_scalar = sensel.cuda()

            contact_pred = (force_pred_scalar > config.CONTACT_THRESH).long()
            contact_gt = (force_gt_scalar > config.CONTACT_THRESH).long()
            any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
            any_contact_gt = torch.sum(contact_gt, dim=(1, 2)) > 0

            force_pred = ft_pred['bottleneck_logits'][:, :3]
            force_gt = ft[:, :3]
            torque_pred = ft_pred['bottleneck_logits'][:, 3:]
            torque_gt = ft[:, 3:]

            any_ft_contact_gt_1N = torch.norm(force_gt, dim=1) > 1
            any_ft_contact_gt_2N = torch.norm(force_gt, dim=1) > 2
            any_ft_contact_gt_3N = torch.norm(force_gt, dim=1) > 3

            # weak_val_metrics['contact_accuracy'](any_contact_pred, any_contact_gt)
            weak_val_metrics['temporal_accuracy_1N'](any_contact_pred, any_ft_contact_gt_1N)
            weak_val_metrics['temporal_accuracy_2N'](any_contact_pred, any_ft_contact_gt_2N)
            weak_val_metrics['temporal_accuracy_3N'](any_contact_pred, any_ft_contact_gt_3N)
            # val_metrics['contact_f1_1N'](any_contact_pred, any_contact_gt_1N)
            # val_metrics['contact_f1_1N'](any_contact_pred, any_ft_contact_gt_1N)
            # val_metrics['contact_f1_2N'](any_contact_pred, any_ft_contact_gt_2N)
            # val_metrics['contact_f1_3N'](any_contact_pred, any_ft_contact_gt_3N)
            weak_val_metrics['force_rmse'](force_pred, force_gt)
            weak_val_metrics['torque_rmse'](torque_pred, torque_gt)

            # we only want to compute temporal accuracy on the frames where the ground truth force is not within the deadband
            # ignore index is 2

            # any_contact_gt_deadband_1_3 is false when < 1N, true when > 3N, and 2 when in the deadband
            any_contact_gt_deadband_1_3 = 2 * torch.ones_like(any_ft_contact_gt_1N)
            any_contact_gt_deadband_1_3[torch.norm(force_gt, dim=1) < 1] = 0
            any_contact_gt_deadband_1_3[torch.norm(force_gt, dim=1) > 3] = 1

            # any_contact_gt_deadband_2_3 is false when < 2N, true when > 3N, and 2 when in the deadband
            any_contact_gt_deadband_2_3 = 2 * torch.ones_like(any_ft_contact_gt_2N)
            any_contact_gt_deadband_2_3[torch.norm(force_gt, dim=1) < 2] = 0
            any_contact_gt_deadband_2_3[torch.norm(force_gt, dim=1) > 3] = 1

            weak_val_metrics['temporal_accuracy_deadband_1_3'](any_contact_pred, any_contact_gt_deadband_1_3)
            weak_val_metrics['temporal_accuracy_deadband_2_3'](any_contact_pred, any_contact_gt_deadband_2_3)
    
 
    print('WEAKLY_LABELED_TEST')
    print_metrics(weak_val_metrics, config, checkpoint_path)

def evaluate(config, force_test_on_test=False, force_zero_guesser=False):
    evaluate_labeled(config)
    evaluate_weak(config)

if __name__ == "__main__":
    config, args = parse_config_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate(config)