import torch
import numpy as np
import segmentation_models_pytorch as smp
from utils.pred_utils import *
from prediction.model.fpn_bottleneck_logits_model import FPNBottleneckLogits
from prediction.model.fpn_dann_model import FPN_DANN
from prediction.model.fpn_dann_logits_model import FPN_DANN_Logits
from prediction.model.fpn_dann_logits_model_regression import FPN_DANN_Logits_Regression
from prediction.model.masked_binary_cross_entropy import MaskedBCELoss
from prediction.model.soft_cross_entropy import SoftCrossEntropyLoss
from prediction.model.log_mse import LogMSELoss
from prediction.model.weighted_mse import WeightedMSELoss
from prediction.model.fpn_cdan_logits_model import FPN_CDAN_Logits
from prediction.loader import WerpData

def build_model(config, device, phases):
    out_dict = dict()

    if config.FORCE_CLASSIFICATION:
        if hasattr(config, 'USE_DICE_LOSS') and config.USE_DICE_LOSS:
            out_dict['criterion'] = smp.losses.DiceLoss('multiclass')
        elif hasattr(config, 'USE_JACCARD_LOSS') and config.USE_JACCARD_LOSS:
            out_dict['criterion'] = smp.losses.JaccardLoss('multiclass')
        elif hasattr(config, 'USE_SOFT_CROSS_ENTROPY') and config.USE_SOFT_CROSS_ENTROPY:
            weight = [float(config.FORCE_CLASSIFICATION_NONZERO_WEIGHT)] * config.NUM_FORCE_CLASSES
            weight[0] = 1
            out_dict['criterion'] = SoftCrossEntropyLoss(omega=config.SOFT_CROSS_ENTROPY_OMEGA, num_classes=config.NUM_FORCE_CLASSES, weight=torch.tensor(weight).cuda())
        else:   # normal cross-entropy, weighted in this case
            weight = [float(config.FORCE_CLASSIFICATION_NONZERO_WEIGHT)] * config.NUM_FORCE_CLASSES
            weight[0] = 1
            out_dict['criterion'] = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())

        out_channels = config.NUM_FORCE_CLASSES
    else:
        if hasattr(config, 'USE_MAE') and config.USE_MAE:
            out_dict['criterion'] = torch.nn.L1Loss()
        elif hasattr(config, 'USE_LOG_MSE') and config.USE_LOG_MSE:
            out_dict['criterion'] = LogMSELoss()
        else:
            out_dict['criterion'] = torch.nn.MSELoss()
        out_channels = 1

    if hasattr(config, 'USE_NONMASKED_LOSS') and config.USE_NONMASKED_LOSS:
        # out_dict['criterion_weak'] = torch.nn.BCELoss()
        print('USING WEIGHTED MSE WEAK LOSS')
        out_dict['criterion_weak'] = WeightedMSELoss(force_weight=1.0, torque_weight=config.LOSS_RATIO)
    else:
        # out_dict['criterion_weak'] = MaskedBCELoss()
        print('MASKED LOSS NOT IMPLEMENTED YET')
    out_dict['criterion_dann'] = torch.nn.CrossEntropyLoss()

    print('Loss function:', out_dict['criterion'])
    num_weak_logits = 6
    if hasattr(config, 'WEAK_LABEL_HIGH_LOW') and config.WEAK_LABEL_HIGH_LOW:
        num_weak_logits = 7
    print('Number of weak logits:', num_weak_logits)

    if config.NETWORK_TYPE == 'smp':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn = resnet_preprocessor
    elif config.NETWORK_TYPE == 'smp_stacked_bottleneck':
        ENCODER = 'se_resnext50_32x4d_stacked'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn = resnet_preprocessor
    elif config.NETWORK_TYPE == 'smp_unet':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    elif config.NETWORK_TYPE == 'smp_deeplab':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'fpn_bottleneck_logits':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = FPNBottleneckLogits(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS,
            num_out_logits=num_weak_logits
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'fpn_dann':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = FPN_DANN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'fpn_dann_logits':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = FPN_DANN_Logits(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS,
            num_out_logits=num_weak_logits
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'fpn_dann_logits_regression':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create regression model with pretrained encoder
        model = FPN_DANN_Logits_Regression(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS,
            num_out_logits=num_weak_logits
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'fpn_cdan_logits':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = FPN_CDAN_Logits(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS,
            num_out_logits=num_weak_logits,
            use_conv_compression=config.USE_CONV_COMPRESSION
        )

        preprocessing_fn = resnet_preprocessor

    elif config.NETWORK_TYPE == 'pytorch_deeplab':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    else:
        raise ValueError('Unknown model')

    model = model.to(device)

    out_dict['model'] = model

    if 'train' in phases:
        override_force_path = None
        if hasattr(config, 'OVERRIDE_FORCE_PATH'):
            override_force_path = config.OVERRIDE_FORCE_PATH

        out_dict['train_dataset'] = WerpData(folder=config.TRAIN_FOLDER, stage='train')

    # if 'source_domain' in phases:
    #     out_dict['source_domain_dataset'] = ForceDataset(config, config.SOURCE_DOMAIN_FILTER,
    #                                              image_method=config.DATALOADER_IMAGE_METHOD,
    #                                              force_method=config.DATALOADER_FORCE_METHOD,
    #                                              skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES,
    #                                              preprocessing_fn=preprocessing_fn,
    #                                              phase='train')

    if 'target_domain' in phases:
        out_dict['target_domain_dataset'] = WerpData(folder=config.WEAK_TRAIN_FOLDER, stage='train')

    if 'val' in phases:
        # include_raw_force is true if we're not training
        out_dict['val_dataset'] = WerpData(folder=config.TEST_FOLDER, stage='test')

    if 'val_weak' in phases:
        out_dict['val_weak_dataset'] = WerpData(folder=config.WEAK_TEST_FOLDER, stage='test')

    if 'test' in phases:
        raise ValueError('NO TESTING YET!!!')

    if hasattr(config, 'STRONGWEAK_EXPERIMENT_FULLY_LABELED_PERCENT') and 'train_dataset' in out_dict and 'target_domain_dataset' in out_dict:
        # Overwrite the data in both the train dataset and target dataset
        per_seq_data = out_dict['train_dataset'].per_seq_datapoints
        split_index = int(len(per_seq_data) * config.STRONGWEAK_EXPERIMENT_FULLY_LABELED_PERCENT)
        train_data = per_seq_data[:split_index]
        target_data = per_seq_data[split_index:]
        print('DOING STRONGWEAK EXPERIMENT!!!!!!!!!!!!!!!!!!!')
        print('original len {} new train len {} new target len {} sum len {}'.format(len(per_seq_data), len(train_data), len(target_data), len(train_data) + len(target_data)))

        out_dict['train_dataset'].all_datapoints = [item for sublist in train_data for item in sublist]
        print('New training set size', len(out_dict['train_dataset'].all_datapoints))

        if hasattr(config, 'STRONGWEAK_INCLUDE_TARGET') and config.STRONGWEAK_INCLUDE_TARGET:
            to_extend_points = [item for sublist in target_data for item in sublist]
            print('Extending current target set. Current length {}, extend length {}'.format(len(out_dict['target_domain_dataset'].all_datapoints), len(to_extend_points)))
            out_dict['target_domain_dataset'].all_datapoints.extend(to_extend_points)
            print('New target set length', len(out_dict['target_domain_dataset']))
        else:
            out_dict['target_domain_dataset'].all_datapoints = [item for sublist in target_data for item in sublist]

    return out_dict

