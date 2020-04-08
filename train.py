import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
from datasetloader import FacialKeypointsDataset
from torch.utils.data import Dataset, DataLoader
from model import Net
import torch.nn as nn
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct the dataset
face_dataset = FacialKeypointsDataset(
    csv_file='data/training_frames_keypoints.csv',
    root_dir='data/training/')

test_dataset = FacialKeypointsDataset(
    csv_file='data/test_frames_keypoints.csv',
    root_dir='data/test/')


# load data in batches
batch_size = 32
train_loader = DataLoader(face_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

batch_size = 32
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)


net = Net()

# Define the loss and optimization
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# transfer model to device
net.to(device)

# train your network
n_epochs = 20

net.train()

for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0

    # train on batches of data, assumes you already have train_loader
    for batch_i, data in enumerate(train_loader):
        # get the input images and their corresponding labels
        images = data['image']
        key_pts = data['keypoints']

        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)

        # convert variables to floats for regression loss
        key_pts = key_pts.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)

        # transfer data to device
        images, key_pts = images.to(device), key_pts.to(device)

        # forward pass to get outputs
        output_pts = net(images)

        # calculate the loss between predicted and target keypoints
        loss = criterion(output_pts, key_pts)

        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        # backward pass to calculate the weight gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # print loss statistics
        # convert loss into a scalar and add it to the running_loss
        running_loss += loss.item()
        if batch_i % 10 == 9:    # print every 10 batches
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                epoch + 1, batch_i+1, running_loss/1000))
            running_loss = 0.0
        # Validation Loop
        if batch_i % 100 == 99:
            net.eval()
            test_loss = 0
            accuracy = 0
            for _, data in enumerate(train_loader):
                images = data['image']
                key_pts = data['keypoints']
                key_pts = key_pts.view(key_pts.size(0), -1)
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)
                images, key_pts = images.to(device), key_pts.to(device)
                output_pts = net(images)
                loss = criterion(output_pts, key_pts)
                test_loss += loss.item()
            print('Epoch: {}, Batch: {}, Test. Loss: {}'.format(
                epoch + 1, batch_i+1, test_loss/1000))
            net.train()


print('Finished Training')

model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)
