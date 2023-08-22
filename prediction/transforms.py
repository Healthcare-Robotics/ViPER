import torchvision.transforms as transforms
from utils.config_utils import *
import cv2

class WerpTransforms:
    def __init__(self, transform_type, stage, pixel_mean=0.5, pixel_std=0.5):
        self.config, self.args = parse_config_args()
        self.img_size = (self.config.NETWORK_IMAGE_SIZE_Y, self.config.NETWORK_IMAGE_SIZE_X) # (height, width)
        self.transform_type = transform_type
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.stage = stage
        self.transforms = self.choose_transform(pixel_mean, pixel_std)
    
    def __call__(self, img):
        return self.transforms(img)

    def bottom_center_crop(self, img):
        # resizes the image and takes a bottom center crop such that the image contains the entire gripper
        return transforms.functional.crop(img, top=img.size()[1] - self.img_size[0], left=img.size()[2] // 2 - (self.config.NETWORK_IMAGE_SIZE_X - 25) // 2, height=self.config.NETWORK_IMAGE_SIZE_Y, width=self.config.NETWORK_IMAGE_SIZE_X)
    
    def bottom_center_crop_sensel(self, img):
        pass

    def bottom_center_crop_big(self, img):
        # takes a bottom center crop such that the image contains the entire gripper
        width = 400
        height = 400
        return transforms.functional.crop(img, top=img.size()[1] - height, left=img.size()[2] // 2 - width // 2, height=height, width=width)

    def get_random_crop_transforms(self, pixel_mean, pixel_std):
        # performs bottom center crop and then random crop
        random_crop_transforms = transforms.Compose(
        [
            [transforms.Lambda(self.bottom_center_crop_big),
            transforms.Resize(size=275),
            transforms.RandomCrop(size=self.img_size)]
        ]
        )
        return random_crop_transforms
    
    def get_minimal_transforms(self):
        minimal_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size),
        ]
        )
        return minimal_transforms

    def get_test_transforms(self, pixel_mean, pixel_std):
        custom_transform_list = [transforms.Resize(self.img_size)]

        if 'crop' in self.transform_type:
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop)]
        elif 'random_crop' in self.transform_type:
            # cannot have both crop and random_crop
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop_big), transforms.Resize(size=275), transforms.RandomCrop(size=self.img_size)]
        
        # # UNCOMMENT FOR TEST TIME JITTER
        # if 'jitter' in self.transform_type:
        #     custom_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        
        if 'normalize' in self.transform_type:
            custom_transform_list.append(transforms.Normalize(pixel_mean, pixel_std))
        
        # print('custom_transform_list: ', custom_transform_list)
        return transforms.Compose(custom_transform_list)

    def get_train_transforms(self, pixel_mean, pixel_std):
        custom_transform_list = [transforms.Resize(self.img_size)]

        if 'crop' in self.transform_type:
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop)]
        elif 'random_crop' in self.transform_type:
            # cannot have both crop and random_crop
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop_big), transforms.Resize(size=275), transforms.RandomCrop(size=self.img_size)]

        if 'jitter' in self.transform_type:
            custom_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        elif 'lighting' in self.transform_type:
            custom_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0))
        if 'normalize' in self.transform_type:
            custom_transform_list.append(transforms.Normalize(pixel_mean, pixel_std))
        
        return transforms.Compose(custom_transform_list)

    def get_sensel_transforms(self, pixel_mean, pixel_std):
        custom_transform_list = [transforms.Resize(self.img_size)]

        if 'crop' in self.transform_type:
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop)]
        elif 'random_crop' in self.transform_type:
            # cannot have both crop and random_crop
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop_big), transforms.Resize(size=275), transforms.RandomCrop(size=self.img_size)]

        return transforms.Compose(custom_transform_list)

    def choose_transform(self, pixel_mean, pixel_std):
        if self.stage == 'train':
            if type(self.transform_type) is list:
                return self.get_train_transforms(pixel_mean, pixel_std)
            else:
                raise ValueError('Error: configs.transform should be a list of strings')
        elif self.stage == 'test':
            return self.get_test_transforms(pixel_mean, pixel_std)
        elif self.stage == 'sensel':
            return self.get_sensel_transforms(pixel_mean, pixel_std)