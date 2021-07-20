from PIL import Image
import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,GaussNoise
from albumentations.augmentations.transforms import Cutout,PadIfNeeded
from albumentations.pytorch import ToTensorV2


class album_Compose_train:
    def __init__(self,rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p):
        self.transform = Compose(
        [
         PadIfNeeded(min_height=min_height, min_width=min_width, border_mode=cv2.BORDER_CONSTANT, value=value, p=p),
         RandomCrop(rcorp,rcorp, p=p),
         Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size,  fill_value=value),
         HorizontalFlip(p=flip_p),
         #GaussNoise(p=0.15),
         #ElasticTransform(p=0.15),
        Normalize(n1, (n2)),
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


       
			
def get_train_transform_64():
    rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p = 64,80,80,1,8,8,[0.4914*255, 0.4822*255, 0.4465*255],1.0,(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010),0.2
    transform = album_Compose_train(rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p)
    return transform

def get_train_transform_48():
    rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p = 48,64,64,1,8,8,[0.4914*255, 0.4822*255, 0.4465*255],1.0,(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010),0.2
    transform = album_Compose_train(rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p)
    return transform
	
def get_train_transform_32():
    rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p = 64,80,80,1,8,8,[0.4914*255, 0.4822*255, 0.4465*255],1.0,(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010),0.2
    transform = album_Compose_train(rcorp,min_height,min_width,num_holes,max_h_size,max_w_size,value,p,n1,n2,flip_p)
    return transform


def get_test_transform():
    transform = album_Compose_test()
    return transform