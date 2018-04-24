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

#FC model
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.h1_layer = nn.Sequential(
            nn.Linear(28*28, 100),
            #nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        self.h2_layer = nn.Sequential(
            nn.Linear(100, 10),
            #nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, input):
        output = self.h1_layer(input)
        output = self.h2_layer(output)
        #output = nn.Softmax(output)
        return output

#Model, loss, and optimizer
model = FC()
XE_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Use GPU
model.cuda()

#Train
model.train()
for epoch in range(num_epochs):
    for index,(images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        images = torch.squeeze(images)
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        output = model(images.view(-1,28*28))
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
    images = torch.squeeze(images)
    output = model(images.view(-1,28*28))
    _, predicted = torch.max(output.data, 1)
    num += labels.size(0)
    correct += (predicted == labels).sum()

print("The result of the test is %.2f %%"%(100 * correct/num))

#Save the model
torch.save(model.state_dict(), 'fc_model.pkl')