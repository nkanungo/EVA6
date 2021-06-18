from PIL import Image
import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,GaussNoise, ShiftScaleRotate, CoarseDropout,ToGray
from albumentations.augmentations.transforms import Cutout
from albumentations.pytorch import ToTensorV2


class album_Compose_train:
    def __init__(self):
        self.transform = Compose(
        [
         HorizontalFlip(p=0.1),
         GaussNoise(p=0.1),
         #ElasticTransform(p=0.15),
         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
         CoarseDropout(max_holes=4, max_height=4, max_width=4, min_holes=2, min_height=2, min_width=2, fill_value=0.4914, mask_fill_value=None, always_apply=False, p=0.5),
         Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
         ToGray(always_apply=False, p=0.5),
         ToTensorV2(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class album_Compose_test:
    def __init__(self):
        self.transform = Compose(
        [
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensorV2(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		
		
def get_train_transform():
    transform = album_Compose_train()
    return transform

def get_test_transform():
    transform = album_Compose_test()
    return transform