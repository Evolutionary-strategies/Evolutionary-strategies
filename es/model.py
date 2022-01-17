import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(3, 6, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1176, 128)
        self.fc2 = nn.Linear(128, 10)
        for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # kilde: https://discuss.pytorch.org/t/how-to-manually-set-the-weights-in-a-two-layer-linear-model/45902
    # Ikke testa!
    def set_params(self, params):
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(torch.from_numpy(params[0:81]).reshape(3,3,3,3).float())
            self.conv2.weight = torch.nn.Parameter(torch.from_numpy(params[81:243]).reshape(6,3,3,3).float())
            
            
        
    # kilde: https://discuss.pytorch.org/t/how-to-output-weight/2796
    # Printer, men vet ikke om det er "riktige" tensorer den printer
    def print_layers(self):
        for param in self.parameters():
            print(param.data)

    def save_model(self, name = "example"):
        path = "../models/" + name + ".pt"
        torch.save(self.state_dict(), path)

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

net = Net()

print("Model's state_dict:")
# for param_tensor in net.state_dict():
    #print(param_tensor, "\t", net.state_dict()[param_tensor])

print("val", net.state_dict()['conv1.weight'])

def load_es_model(params, path = "../models/example.pt"):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.set_params(params)
    return net

def load_gd_model(path = "../models/example.pt"):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

"""from prettytable import PrettyTable

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
    
count_parameters(net)"""
