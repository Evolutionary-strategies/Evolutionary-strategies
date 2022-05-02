import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
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
    def __init__(self, requires_grad = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, 128)#???
        self.fc2 = nn.Linear(128, 10)
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def set_params(self, params) -> None:
        c1 = 1728
        c1_b = c1 + 64
        c2 = 73728 + c1_b
        c2_b = c2 + 128
        f1 = 589824 + c2_b
        f1_b = f1 + 128
        f2 = 1280 + f1_b
        f2_b = f2+10

        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(torch.from_numpy(params[0:c1]).reshape(64,3,3,3).float())
            self.conv1.bias = torch.nn.Parameter(torch.from_numpy(params[c1:c1_b]).reshape(64).float())

            self.conv2.weight = torch.nn.Parameter(torch.from_numpy(params[c1_b:c2]).reshape(128,64,3,3).float())
            self.conv2.bias = torch.nn.Parameter(torch.from_numpy(params[c2:c2_b]).reshape(128).float())

            self.fc1.weight = torch.nn.Parameter(torch.from_numpy(params[c2_b:f1]).reshape(128, 4608).float())
            self.fc1.bias = torch.nn.Parameter(torch.from_numpy(params[f1:f1_b]).reshape(128).float())

            self.fc2.weight = torch.nn.Parameter(torch.from_numpy(params[f1_b:f2]).reshape(10, 128).float())
            self.fc2.bias = torch.nn.Parameter(torch.from_numpy(params[f2:f2_b]).reshape(10).float())

            
    def print_layers(self) -> None:
        for param in self.parameters():
            print(param.data)

    def save_model(self, name = "example.pt") -> None:
        model_folder_path = '../models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        name = os.path.join(model_folder_path, name)
        torch.save(self.state_dict(), name)

    def test(self, log=False) -> float:
        loss_fn=nn.CrossEntropyLoss()
        # dataiter = iter(testloader)
        # images, labels = dataiter.next()
        running_loss=0
        correct_tst = 0
        correct_trn = 0
        total_tst = 0
        total_trn = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                loss = loss_fn(outputs,labels)
                running_loss+=loss.item()
                total_tst += labels.size(0)
                correct_tst += (predicted == labels).sum().item()
            for data in trainloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total_trn += labels.size(0)
                correct_trn += (predicted == labels).sum().item()
        test_loss=running_loss/len(testloader)
        acc_tst = correct_tst / total_tst
        acc_trn = correct_trn / total_trn
        if(log):
            logger.info(f'Accuracy: {acc_tst}, Loss: {test_loss:.3f} ')
            logger.info("Train Accuracy: " +  str(acc_trn))
            logger.info("Test Accuracy: " +  str(acc_trn))
        return (acc_tst, acc_trn)

    def set_params_and_test(self, params) -> float:
        self.set_params(params)
        return self.test()


def load_model(path = "../models/example.pt", grad = False) -> Net:
    logger.info("Loading model")
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    for param in net.parameters():
            param.requires_grad = grad
    return net

def train(net, acc_limit = 1) -> None:
    logger.info("Training Gradient decent model")
    torch.set_num_threads(os.cpu_count())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for e, epoch in enumerate(range(5)):  # loop over the dataset multiple times
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
            if i % 100 == 99:    # print every 100 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                if e == 0:
                    logger.info("run: " + str(i))
                else:
                    logger.info("run: " + str(i*e))
                running_loss = 0.0

                if acc_limit < 1:
                    acc, _ = net.test(True)
                    if acc >= acc_limit:
                        logger.info('Finished Training')
                        return
    logger.info('Finished Training')



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