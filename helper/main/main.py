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
from utils.misclassified_image_utils import  display_misclassfied_images
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
#!pip install torch-lr-finder
import torch.optim as optim
from torch_lr_finder import LRFinder



# Perform 70:30 split between training and validation

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def model_summary():
    #!pip install torchsummary
    from torchsummary import summary
    from models.resnet18 import ResNet18
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    summary(model, input_size=(3, 64, 64))
    
def train_valid_split(base_dir,valid_split):
    perform_train_validation_split(	base_dir= base_dir, validation_split = valid_split)
    
def load_transfer(base_dir,batch_size,transform_type):

    if transform_type == 64:    
        transform_train = get_train_transform_64()
    if transform_type == 48:    
        transform_train = get_train_transform_48()
    if transform_type == 32:    
        transform_train = get_train_transform_32()      
        
    transform_test = get_test_transform()
    trainloader = get_train_loader(base_dir, batch_size, transform_train)
    testloader = get_test_loader(base_dir,batch_size, transform_test)
    classes = get_classes(base_dir)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    return trainloader, testloader, classes

def define_network():
    #!pip install torchsummary
    from torchsummary import summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    summary(model, input_size=(3, 64, 64))
    
def lr_finder_exp(lr=0.001,momentum=0.9, weight_decay=0.0001,end_lr=10, num_iter=100,trainloader = None):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader, end_lr=end_lr, num_iter=num_iter)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return criterion
    
def lr_finder_linear(lr=0.01,momentum=0.9, weight_decay=0.0001,end_lr=0.1, num_iter=100,trainloader = None,testloader = None,criterion = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader, val_loader=testloader, end_lr=end_lr, num_iter=num_iter, step_mode="linear")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
    
def train_model(lr=0.03, momentum=0.9, weight_decay=0.0001,EPOCHS = 50,trainloader = None,testloader = None, path=None ):
    from tqdm import tqdm
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.optim as optim
    import os

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18(num_classes=200).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    EPOCHS = EPOCHS
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)


    PATH = path
    torch.save(model.state_dict(), PATH)
    best_test_accuracy = 0.0
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc)
        test(model, device, testloader, test_losses, test_acc)
        t_acc = test_acc[-1]
        if t_acc > best_test_accuracy:
            print("Test Accuracy: " + str(t_acc) + " has increased. Saving the model")
            best_test_accuracy = t_acc
            torch.save(model.state_dict(), PATH)
        
        scheduler.step(t_acc)
    return model
def display_test_data(testloader = None,classes=None):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    return images,labels

def predict(classes=None, images = None,labels = None,model=None):
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) 
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(4)))
def print_test_accuracy(testloader = None):
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (get_test_accuracy(model, testloader, device)))
    
def accuracy_per_class(testloader = None,classes=None):
    class_correct,class_total = get_accuracy_per_class(model, testloader, device, num_classes=len(classes))
    for i in range(len(classes)):
        if class_total[i] > 0.0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
                
def plot_accuracy(train_acc, test_acc):	
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_acc, label="Train Accuracy")
    axs.plot(test_acc, label="Test Accuracy")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()