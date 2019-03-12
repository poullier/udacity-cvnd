## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


        # input 1x224x224
        # (W-F)/S +1
        # (224 - 5)/1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5) #32x220x220
        self.pool1 = nn.MaxPool2d(2, 2) #32x110x110
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3) #64x108x108
        self.pool2 = nn.MaxPool2d(2, 2) #64x54x54
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2) #128x53x53
        self.pool3 = nn.MaxPool2d(2, 2) #128x26x26
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1) #256x26x26
        self.pool4 = nn.MaxPool2d(2, 2) #256x13x13
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256 * 13 * 13, 1000)
        self.drop5 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(self.fc2(x))
        x = self.fc3(x)

        return x


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # the output Tensor for one image, will have the dimensions: (32, 224, 224)
        self.conv1 = nn.Conv2d(1, 32, 5) # output: 32x220x220

        # maxpool layer
        self.pool1 = nn.MaxPool2d(4, 4) # output: 32x55x55

        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2) # output: 64x26x26

        # 64*26*26 filtered/pooled inputs 1000 outputs
        self.fc1 = nn.Linear(64 * 26 * 26, 1000)

        # dropout with p=0.2
        self.drop = nn.Dropout(p=0.2)

        # finally, create 136 keypoints output from the 1000 inputs
        self.fc2 = nn.Linear(1000, 136)



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # one conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        # final output
        return x