import os
import re
import random
import pandas as pd
import numpy as np
import torch
import torchvision



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):    
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
	
def get_classes(base_dir):
    train_base_dir = base_dir + '/train'
    classes = [d for d in os.listdir(train_base_dir) ]
    classes.sort(key=natural_keys)
    return classes
	
	
def perform_train_validation_split(	base_dir, validation_split = 0.3):
    train_base_dir = base_dir + '/train'
    validation_base_dir = base_dir + '/val'
    validation_map_dataframe = pd.read_csv(validation_base_dir + '/val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    validation_map_dataframe.drop(['X', 'Y', 'H', 'W'], axis=1, inplace=True)
    #validation_map_dataframe.head(3)
    for i, row in validation_map_dataframe.iterrows(): 
        class_name = row['Class']
        file_name = row['File']
        image_dir_copy = train_base_dir + '/' +  class_name + "/images"
        if os.path.exists(image_dir_copy) == False:
            os.makedirs(image_dir_copy) 		
        os.system("cp " + validation_base_dir + "/images/" + file_name + " " + image_dir_copy)

    os.system('rm -rf ' + validation_base_dir)	  

    if os.path.exists(validation_base_dir) == False:
        os.makedirs(validation_base_dir) 

    classes = [d for d in os.listdir(train_base_dir) ]
    classes.sort(key=natural_keys)

    for image_dir in classes:
        #print("image_dir: ", image_dir)
        train_image_dir = train_base_dir + "/" + image_dir + "/images"
        validation_image_dir = validation_base_dir + "/" + image_dir + "/images"
        if os.path.exists(validation_image_dir) == False:
            os.makedirs(validation_image_dir) 
        image_files = [d for d in os.listdir(train_image_dir) ]
        #print("image_files: ", image_files)
        total_images_files = len(image_files)
        #print("total_images_files =", total_images_files )
        sample_size = int(np.floor(total_images_files * validation_split))
        #print("sample_size =", sample_size )
        sample_image_files = random.sample(image_files, sample_size)
        for image in sample_image_files:
            train_image_path = os.path.join(train_image_dir, image)
            validation_image_path = os.path.join(validation_image_dir, image)
            os.system("cp " +  train_image_path + "  " +  validation_image_path )
            os.remove(train_image_path)
			
def get_train_loader(base_dir, batch_size, transform): 
    train_dir = base_dir + '/train'
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainloader	
	
def get_test_loader(base_dir, batch_size, transform): 
    test_dir = base_dir + '/val'
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
 							
    return testloader