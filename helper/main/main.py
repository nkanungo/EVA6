from PIL import Image
import cv2
import os
import re
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from data_transformations.data_transform_tiny_imagenet_resnet import get_train_transform_64, get_train_transform_48, get_train_transform_32, get_test_transform
from data_loaders.tiny_imagenet_data_loader import perform_train_validation_split, get_classes,get_train_loader, get_test_loader
from models.resnet18 import ResNet18
from utils.train_test_utils import train,test
from utils.accuracy_utils import get_test_accuracy,get_accuracy_per_class
from utils.plot_metrics_utils import plot_accuracy
from utils.misclassified_image_utils import  display_misclassfied_ciphar10_images
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchsummary import summary




# Perform 70:30 split between training and validation

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def model_summary():
    !pip install torchsummary
    from torchsummary import summary
    from models.resnet18 import ResNet18
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    summary(model, input_size=(3, 64, 64))
    
def main():
    perform_train_validation_split(	base_dir= './tiny-imagenet-200', validation_split = 0.3)
    transform_train = get_train_transform_64()
    transform_test = get_test_transform()
    trainloader = get_train_loader('./tiny-imagenet-200', 256, transform_train)
    testloader = get_test_loader('./tiny-imagenet-200',256, transform_test)
    classes = get_classes('./tiny-imagenet-200')
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    
    