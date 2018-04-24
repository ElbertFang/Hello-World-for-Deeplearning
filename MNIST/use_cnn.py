from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

num_epochs = 5
batch_size = 100
learning_rate = 0.001
num_workers = 3

#Download the MNIST Dataset (it will not download if the data has been downloaded before)
train_data = datasets.MNIST('./data', train = True, transform = transforms.ToTensor(), download = True)
test_data = datasets.MNIST('./data', train = False, transform = transforms.ToTensor(), download = True)

#Date Loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)

#CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#Model, loss, and optimizer
model = CNN()
XE_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Use GPU
model.cuda()

#Train
model.train()
for epoch in range(num_epochs):
    for index,(images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        #images = torch.squeeze(images)
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = XE_loss(output, labels)
        loss.backward()
        optimizer.step()

        #if(index+1 % 5 == 0):
    print("Epoch[%d/%d]  XE_Loss:%.5f"
                  %(epoch+1, num_epochs,  loss.data[0]))

#Test
model.eval()
num = 0
correct = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    labels = labels.cuda()
    #images = torch.squeeze(images)
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    num += labels.size(0)
    correct += (predicted == labels).sum()

print("The result of the test is %.2f %%"%(100 * correct/num))

#Save the model
torch.save(model.state_dict(), 'cnn_model.pkl')