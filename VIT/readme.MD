Developers
=============

Monimoy Purkayastha (monimoyd@gmail.com)

Nihar Kanungo (nihar.kanungo@gmail.com)


Processing
===========

1.	This blog implements the Visual Transformers on a Kaggle Dataset which contains images of Cats and Dogs.
2.	Initially We need to install the ViT PyTorch package and Linformer 
3.	Define the Hyper Parameters (Batch Size, Epochs and Learning Rate)
4.	Download the dataset from Kaggle 
5.	Unzip the file to the Colab 
6.	Then we split the data into Training Data and Validation Data
7.	Define Transformation Functions to transform the images (Resize, Crop, Flip)
8.	Apply Transformation to the Training data
9.	Loading Data loader for Training, Validation and Test 
10.	Define Line Transformer which contains Dimension, Sequence length, Depth, Number of Heads
11.	Define the Visual Transformer model which takes the input as dimension, image size, patch size, number of classes and channels 
12.	Define the Loss function and Optimizer 
13.	In this program we use Step Learning Rate
14.	We start Training the model using the data and hyper parameters for 20 epochs and found the accuracy to be 67.5% 


Log
======
HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 15 - loss : 0.6064 - acc: 0.6628 - val_loss : 0.6075 - val_acc: 0.6580

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 16 - loss : 0.5969 - acc: 0.6718 - val_loss : 0.5903 - val_acc: 0.6831

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 17 - loss : 0.5948 - acc: 0.6731 - val_loss : 0.5872 - val_acc: 0.6822

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 18 - loss : 0.5899 - acc: 0.6763 - val_loss : 0.5904 - val_acc: 0.6830

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 19 - loss : 0.5890 - acc: 0.6770 - val_loss : 0.5878 - val_acc: 0.6883

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 20 - loss : 0.5873 - acc: 0.6783 - val_loss : 0.5963 - val_acc: 0.6750 

