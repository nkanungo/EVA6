import torch
import torch.nn as nn
import torch.nn.functional as F


# Normalization Function 
'''
This is a function which takes many parameters.
type: The user is expected to pass one of the three values 

     - BN : If user pass this then the program must perform Batch Normalization

     - LN : If user pass this then the program must perform Layer Normalization

     - If none of the above is passed then the program must perform Group Normalization

How this change affects the Program
-------------------------------------
We have defined the network here . Usually we code Batch normalization directly in the network . However due to this generalization , the code will call function after performing the convolution operation . Based on the type passed by the user this function will return the code to perform the specific normalization. 

'''
def perform_norm(type, num_channels, channel_size_w, channel_size_h, num_groups=2):
    if type == 'BN':
        return nn.BatchNorm2d(num_channels)
    elif type == 'LN':
        return nn.LayerNorm((num_channels, channel_size_w, channel_size_h))
    elif type == 'GN':
        return nn.GroupNorm(num_groups, num_channels)


class Net(nn.Module):
    def __init__(self, type):
        super(Net, self).__init__()
        dropout_prob=0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            perform_norm(type, 12, 28, 28, 2),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            perform_norm(type, 12, 28, 28, 2),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            perform_norm(type, 14, 14, 14, 2),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            perform_norm(type, 14, 14, 14, 2),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            perform_norm(type, 14, 5, 5, 2),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)