import torch
import numpy as np
from prediction.loader import *
from torch.utils.tensorboard import SummaryWriter
import utils.recording_utils as util
from utils.pred_utils import *
import argparse
import datetime
from prediction.model_builder import build_model
import prediction.evaluator as evaluator
from tqdm import tqdm
from prediction.loader import WerpData
from torch.utils.data import DataLoader
import wandb

def val_epoch(config, model_dict, val_loader, val_metrics, epoch):
    model = model_dict['model']
    criterion = model_dict['criterion']

    model.eval()
    loss_meter = util.AverageMeter('Loss', ':.4e')
    evaluator.reset_metrics(val_metrics)

    with torch.no_grad():
        # for idx, data in enumerate(tqdm(val_loader)):
        for img, undistorted_img, ft, states, sensel, cam_params in tqdm(val_loader):

            # # view img and sensel for debugging
            # sensel_viewable = pressure_to_colormap(classes_to_scalar(sensel[0].numpy().astype(np.uint8), config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)
            # img_viewable = undistorted_img[0].permute(1, 2, 0).numpy()
            # img_viewable = (img_viewable * 255).astype(np.uint8)

            # cv2.imshow('gt_overlay_val', cv2.addWeighted(img_viewable, 1.0, sensel_viewable, 1.0, 0.0))
            # cv2.waitKey(0)

            image_gpu = undistorted_img.cuda()
            force_gt_gpu = sensel.cuda()
            batch_size = undistorted_img.shape[0]
            ft = ft.cuda()

            force_estimated, ft_pred = model(image_gpu, alpha=0)
            loss = criterion(force_estimated, force_gt_gpu)

            loss_meter.update(loss.item(), batch_size)

            if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(force_estimated, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = force_estimated.squeeze(1) * config.NORM_FORCE_REGRESS

            force_gt_scalar = force_gt_gpu
            evaluator.run_metrics(val_metrics, force_gt_scalar, force_pred_scalar, ft, ft_pred, config)

    wandb.log({'val/loss': loss_meter.avg})
    for key, metric in val_metrics.items():
        wandb.log({'val/' + key: metric.compute()})

    print('Finished val epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_meter.avg))

def val_weak_epoch(config, model_dict, val_weak_loader, val_metrics, epoch):
    model = model_dict['model']
    model.eval()
    evaluator.reset_metrics(val_metrics)

    with torch.no_grad():
        for img, undistorted_img, ft, states, sensel, cam_params in tqdm(val_weak_loader):
            undistorted_img = undistorted_img.cuda()
            weak_labels_source_gt = ft.cuda()
            ft = ft.cuda()

            force_estimated, ft_pred = model(undistorted_img, alpha=0)

            if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(force_estimated, dim=1)
                force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = force_estimated.squeeze(1) * config.NORM_FORCE_REGRESS

            contact_pred = (force_pred_scalar > config.CONTACT_THRESH).long()
            any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0

            force_pred = ft_pred['bottleneck_logits'][:, :3]
            force_gt = ft[:, :3]
            torque_pred = ft_pred['bottleneck_logits'][:, 3:]
            torque_gt = ft[:, 3:]

            any_ft_contact_gt_1N = torch.norm(force_gt, dim=1) > 1
            any_ft_contact_gt_2N = torch.norm(force_gt, dim=1) > 2
            any_ft_contact_gt_3N = torch.norm(force_gt, dim=1) > 3

            # val_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)
            val_metrics['temporal_accuracy_1N'](any_contact_pred, any_ft_contact_gt_1N)
            val_metrics['temporal_accuracy_2N'](any_contact_pred, any_ft_contact_gt_2N)
            val_metrics['temporal_accuracy_3N'](any_contact_pred, any_ft_contact_gt_3N)
            # val_metrics['contact_f1_1N'](any_contact_pred, any_contact_gt_1N)
            # val_metrics['contact_f1_1N'](any_contact_pred, any_ft_contact_gt_1N)
            # val_metrics['contact_f1_2N'](any_contact_pred, any_ft_contact_gt_2N)
            # val_metrics['contact_f1_3N'](any_contact_pred, any_ft_contact_gt_3N)
            val_metrics['force_rmse'](force_pred, force_gt)
            val_metrics['torque_rmse'](torque_pred, torque_gt)

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

            val_metrics['temporal_accuracy_deadband_1_3'](any_contact_pred, any_contact_gt_deadband_1_3)
            val_metrics['temporal_accuracy_deadband_2_3'](any_contact_pred, any_contact_gt_deadband_2_3)

    for key, metric in val_metrics.items():
        wandb.log({'val_weak/' + key: metric.compute()})

    print('Finished val weak epoch: {}. Force RMSE: {:.4f} --------------------'.format(epoch, val_metrics['force_rmse'].compute()))
    print('--------------------------- Torque RMSE: {:.4f} --------------------'.format(val_metrics['torque_rmse'].compute()))

def train_epoch(config, model_dict, source_domain_loader, target_domain_loader, optimizer, scheduler, epoch):
    model = model_dict['model']
    criterion = model_dict['criterion']
    criterion_dann = model_dict['criterion_dann']
    criterion_weak = model_dict['criterion_weak']

    model.train()
    loss_image_meter = util.AverageMeter('Loss image', ':.4e')
    loss_domain_source_meter = util.AverageMeter('Loss domain source', ':.4e')
    loss_domain_target_meter = util.AverageMeter('Loss domain target', ':.4e')
    loss_logits_source_meter = util.AverageMeter('Loss logits source', ':.4e')
    loss_logits_target_meter = util.AverageMeter('Loss logits target', ':.4e')

    iterations = 0
    global global_iter
    global_iter = 0

    source_domain_iterator = iter(source_domain_loader)
    target_domain_iterator = iter(target_domain_loader)

    p = epoch / config.NUM_EPOCHS   # [0-1), how far along with training are we
    alpha = 2. / (1 + np.exp(-10 * p)) - 1  # the gradient reversal weight
    print('Alpha', alpha)

    with tqdm(total=config.TRAIN_ITERS_PER_EPOCH) as progress_bar:
        while iterations < config.TRAIN_ITERS_PER_EPOCH:
            try:
                img_src, undist_img_src, ft_src, states_src, sensel_src, cam_params_src = next(source_domain_iterator)
            except StopIteration:
                print('Stopping iteration source')
                source_domain_iterator = iter(source_domain_loader)
                img_src, undist_img_src, ft_src, states_src, sensel_src, cam_params_src = next(source_domain_iterator)

            try:
                img_tgt, undist_img_tgt, ft_tgt, states_tgt, sensel_tgt, cam_params_tgt = next(target_domain_iterator)
            except StopIteration:
                print('Stopping iteration target')
                target_domain_iterator = iter(target_domain_loader)
                img_tgt, undist_img_tgt, ft_tgt, states_tgt, sensel_tgt, cam_params_tgt = next(target_domain_iterator)

            image_source_gpu = img_src.cuda()
            image_target_gpu = img_tgt.cuda()
            undist_image_source_gpu = undist_img_src.cuda()
            undist_image_target_gpu = undist_img_tgt.cuda()
            force_gt_source_gpu = sensel_src.cuda()
            batch_size_source = img_src.shape[0]
            batch_size_target = img_tgt.shape[0]
            weak_labels_source_gt = ft_src.cuda()
            weak_labels_target_gt = ft_tgt.cuda()

            domain_source_gt = torch.zeros(batch_size_source, dtype=torch.long, device=undist_image_source_gpu.device)
            domain_target_gt = torch.ones(batch_size_target, dtype=torch.long, device=undist_image_target_gpu.device)

            force_estimated_source, dict_source_est = model(undist_image_source_gpu, alpha=alpha)
            force_estimated_target, dict_target_est = model(undist_image_target_gpu, alpha=alpha)

            force_estimated_source = force_estimated_source.squeeze(1)
            force_estimated_target = force_estimated_target.squeeze(1)

            # # view img and sensel
            # sensel_viewable = pressure_to_colormap(classes_to_scalar(sensel_src[0].numpy().astype(np.uint8), config.FORCE_THRESHOLDS), colormap=cv2.COLORMAP_INFERNO)
            # img_viewable = undist_img_src[0].permute(1, 2, 0).numpy()
            # img_viewable = (img_viewable * 255).astype(np.uint8)

            # cv2.imshow('gt_overlay_train', cv2.addWeighted(img_viewable, 1.0, sensel_viewable, 1.0, 0.0))
            # cv2.waitKey(0)

            loss_pressure = criterion(force_estimated_source, force_gt_source_gpu) * config.LAMBDA_PRESSURE
            loss_domain_source = criterion_dann(dict_source_est['domain_logits'], domain_source_gt) * config.LAMBDA_DOMAIN
            loss_domain_target = criterion_dann(dict_target_est['domain_logits'], domain_target_gt) * config.LAMBDA_DOMAIN
            loss_logits_source = criterion_weak(dict_source_est['bottleneck_logits'], weak_labels_source_gt) * config.LAMBDA_WEAK_SOURCE
            loss_logits_target = criterion_weak(dict_target_est['bottleneck_logits'], weak_labels_target_gt) * config.LAMBDA_WEAK_TARGET

            # print('Loss pressure', loss_pressure.item())
            # print('Loss domain source', loss_domain_source.item())
            # print('Loss domain target', loss_domain_target.item())
            # print('Loss logits source', loss_logits_source.item())
            # print('Loss logits target', loss_logits_target.item())

            loss = loss_pressure + loss_domain_source + loss_domain_target + loss_logits_source + loss_logits_target

            loss_image_meter.update(loss_pressure.item(), batch_size_source)
            loss_domain_source_meter.update(loss_domain_source.item(), batch_size_source)
            loss_domain_target_meter.update(loss_domain_target.item(), batch_size_source)
            loss_logits_source_meter.update(loss_logits_source.item(), batch_size_source)
            loss_logits_target_meter.update(loss_logits_target.item(), batch_size_source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += batch_size_source
            global_iter += batch_size_source
            progress_bar.update(batch_size_source)     # Incremental update
            progress_bar.set_postfix(loss=str(loss_image_meter))
            if iterations >= config.TRAIN_ITERS_PER_EPOCH:
                break

    wandb.log({'training/loss_image': loss_image_meter.avg,
                'training/loss_domain_source': loss_domain_source_meter.avg,
                'training/loss_domain_target': loss_domain_target_meter.avg,
                'training/loss_logits_source': loss_logits_source_meter.avg,
                'training/loss_logits_target': loss_logits_target_meter.avg,
                'training/alpha': alpha,
                'training/lr': scheduler.get_last_lr()[0]
    })
    
    print('Finished training epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_image_meter.avg))
    # writer.flush()

def main():
    config, args = parse_config_args()

    with wandb.init(config=wandb.config):
        
        if args.sweep:
            for config_name in ['BATCH_SIZE',
                                'NUM_WORKERS',
                                'TRAIN_ITERS_PER_EPOCH',
                                'NUM_EPOCHS',
                                'LAMBDA_PRESSURE',
                                'LAMBDA_DOMAIN',
                                'LAMBDA_WEAK_SOURCE'
                                'LAMBDA_WEAK_TARGET'
                                ]:

                setattr(config, config_name, getattr(wandb.config, config_name)) # overwrite yaml config values with wandb config values

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_dict = build_model(config, device, ['train', 'target_domain', 'val', 'val_weak'])
        criterion = model_dict['criterion']
        criterion_dann = model_dict['criterion_dann']
        criterion_weak = model_dict['criterion_weak']
        model = model_dict['model']

        if args.use_latest:
            checkpoint_path = find_latest_checkpoint(args.config)
            print('LOADING CHECKPOINT FROM:', checkpoint_path)
            model = torch.load(checkpoint_path)

        print('using batch size', config.BATCH_SIZE)
        print('using num workers', config.NUM_WORKERS)

        source_domain_loader = DataLoader(model_dict['train_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        target_domain_loader = DataLoader(model_dict['target_domain_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        val_weak_loader = DataLoader(model_dict['val_weak_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.LEARNING_RATE_INITIAL)])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LEARNING_RATE_SCHEDULER_STEP,
                                                        gamma=config.LEARNING_RATE_SCHEDULER_GAMMA)

        val_metrics = evaluator.setup_metrics(device)

        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)

        # number of files in ./checkpoints that contain args.config
        folder_index = len([f for f in os.listdir(config.MODEL_DIR) if f.startswith(args.config)])

        if not os.path.exists(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index))):
            os.makedirs(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index)))

        wandb.run.name = '{}_{}'.format(args.config, folder_index)

        if args.skip_val:
            print('SKIPPING VALIDATION UNTIL END OF TRAINING')

        for epoch in range(config.NUM_EPOCHS):
            train_epoch(config, model_dict, source_domain_loader, target_domain_loader, optimizer, scheduler, epoch)

            if not args.skip_val or (args.skip_val and epoch == config.NUM_EPOCHS - 1): # only do val on last epoch if skip_val
                val_epoch(config, model_dict, val_loader, val_metrics, epoch)
                val_weak_epoch(config, model_dict, val_weak_loader, val_metrics, epoch)

            model_name = '{}_{}/model_{}.pth'.format(args.config, folder_index, epoch)
            model_path = os.path.join(config.MODEL_DIR, model_name)

            torch.save(model, model_path)
            print('Saved model to: {}'.format(model_path))
            scheduler.step()
            print('\n')

if __name__ == '__main__':
    config, args = parse_config_args()

    sweep_config = {
        'method': 'grid',
            }
    metric = {
            'name': 'val_loss',
            'goal': 'minimize'
        }
    sweep_config['metric'] = metric
    parameters_dict = {
        'BATCH_SIZE': {
            'values': [12]
        },
        'NUM_WORKERS': {
            'values': [4]
        },
        'TRAIN_ITERS_PER_EPOCH': {
            'values': [10000]
        },
        'NUM_EPOCHS': {
            'values': [10]
        },
        'LAMBDA_PRESSURE': {
            'values': [1]
        },
        'LAMBDA_DOMAIN': {
            'values': [1e-3, 1e-2, 1e-1]
        },
        'LAMBDA_WEAK_SOURCE': {
            'values': [1e-3]
        },
        'LAMBDA_WEAK_TARGET': {
            'values': [1e-3, 1e-2, 1e-1]
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='werp')

    if args.sweep: # wandb configs override yaml configs
        wandb.agent(sweep_id, main) 
    else: # use just yaml configs
        main()