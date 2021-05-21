


Data Representation
=======================
MNIST Dataset consists of 60k Images of 10 classes. The Images are of 28 x 28 in size. 
The Addition of the Random numbers from 0-9 with the MNIST data was made using One hot vector representation of the same batch size 
The Data was converted as torch.tensor before being fed to the training function
The MNISt data was fed in gray scale 

Data Generation 
==================
MNIST Data was loaded directly to the Colab Notebook
The Random data was generated using torch.randint function
Dataset loader was used to load image in batches of defined size .
We tried with custom data loader but finally added the data loader provided by pytorch 
How the Input data were combined

How data sets are combined
============================
MNIST Data Processing

i) Conv1 - 3 x 3 kernel - Target Image size - 26 x 26

ii) Conv2 - 3 x 3 kernel - Target Image Size - 24 x 24

iii) Maxpool2D - Stride -2 - Target Image size - 12 x 12

iv ) Conv 3 - 5 x 5 kernel - Target Image Size - 8 x 8

v) Maxpool2D stride2 - Target Image size 4 x 4

vi) Fully Connected layer FC1, FC2

vii) Output Layer

Random Data + MNIST

i) Concatenate the Random Number after FC2 Layer - Now the Targets are 20

ii) Pass the data through FC3 Layer

iii) Output layer

Evaluation of the result 
==============================
1.	Evaluation of the result by Loss 

The Model calculates the loss based on the difference between predicted label and actual ground truth.
We calculated the loss for the MNIST dataset 
Then we Calculated the loss for the MNIST + Random Dataset 
We then performed two backprop using both the loss. But that didn’t work out
So, we added both the losses and performed backpropagation 

2.	Evaluation of Result by Accuracy

The model predicts the number of correct predictions. Based on this value it calculates the correct prediction divided by the total which provides the accuracy of the model (Training accuracy)

The holdout dataset can be testes with the model to find out validation accuracy

Final Result
================
The MNIST Dataset Accuracy came around 99.41% whereas the MNIST+ Random Number came around 97.12% 
I think with more optimization it may improve further .

Loss Function
===============

We used Cross Entropy Loss function which is also known as Log loss. The Loss function is very simple which calculates the probability of each class and compares it with the actual one. Then based on the difference it penalizes the data based on how far it’s from the expected value.
This loss function is simple and fast. Knowing that MNIST dataset is simple in nature we choose to go with this. We can choose more complex loss functions for complex datasets.
 
 ![](./images/cp.png)

