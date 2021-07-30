
Assignment 12 – EVA 6
=======================
• Developers
=================

Monimoy Purkayastha (monimoyd@gmail.com)
Nihar Kanungo (nihar.kanungo@gmail.com)

Description
==============

This Assignment talks about the Spatial Transformer Network. This concept is based on Attention Mechanism. This network is based on the CIFAR10 database which allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model.

It has the following steps 

1. It loads the CIFAR10 data and apply some augmentation to it 
2. Then it defines a network for 
    a) Special Transformation Localization
    b) Regressor for the Affine Matrix 
    c) Spatial Transformer Forward Network
    d) Transform Input forward function 
    
3. Then we train the model using the data for 50 epochs 
4. Then we test the model to find out the accuracy of Spatial Transofrmer
5. Then we visualize to see the original input and translated output 

More Hyperparameter tunings can be performed to get a better reuslt 
