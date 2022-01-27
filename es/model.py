import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from util import load_data
import os
torch.set_num_threads(1)
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import numpy as np

# Loading data:
loader = load_data()
testloader = loader[1]
trainloader = loader[0]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, 128)#???
        self.fc2 = nn.Linear(128, 10)
        """for param in self.parameters():
            param.requires_grad = False"""

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def set_params(self, params):
        c1 = 1728
        c2 = 73728 + c1
        f1 = 589824 + c2
        f2 = 1280 + f1

        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(torch.from_numpy(params[0:c1]).reshape(64,3,3,3).float())
            self.conv2.weight = torch.nn.Parameter(torch.from_numpy(params[c1:c2]).reshape(128,64,3,3).float())
            self.fc1.weight = torch.nn.Parameter(torch.from_numpy(params[c2:f1]).reshape(128, 4608).float())
            self.fc2.weight = torch.nn.Parameter(torch.from_numpy(params[f1:f2]).reshape(10, 128).float())
            
            
    def print_layers(self):
        for param in self.parameters():
            print(param.data)

    def save_model(self, name = "example"):
        path = "../models/" + name + ".pt"
        torch.save(self.state_dict(), path)
        logger.info("saved model")

    def test(self):
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {correct / total} ')
        return correct / total




def load_model(path = "../models/example.pt"):
    print("Loading model")
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def train(net):
    print("Training Gradient decent model")
    torch.set_num_threads(os.cpu_count())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')



'''
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
net = Net()   
count_parameters(net)'''