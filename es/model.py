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
    
    
    #Treningsmetoder til GD:

    # Predictor
    def f(self, x):
        return torch.softmax(self.forward(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.forward(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())



#Utesta
def load_es_model(params, path = "../models/example.pt"):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.set_params(params)
    return net

#Utesta
def load_gd_model(path = "../models/example.pt"):
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


# funker ikke :((
def train_gd_model():
    print("Training Gradient decent model")
    # torch.set_num_threads(7) #Endre på denne?

    cifar_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
    x_train = torch.tensor(cifar_data.data.reshape(-1, 3, 32, 32)).float()
    y_train = torch.zeros((len(cifar_data.targets), 10))
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    x_test = torch.tensor(cifar_test.data.reshape(-1, 3, 32, 32)).float()
    y_test= torch.zeros((len(cifar_test.targets), 10))

    print(x_train.shape)
    print(y_train.shape)
    print()
    print(x_test.shape)
    print(y_test.shape)
    print()

    batches = 500
    x_train_batches = torch.split(x_train, batches)
    y_train_batches = torch.split(y_train, batches)

    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), 0.05)

    for epoch in range(10):
        for batch in range(len(x_train_batches)):
            net.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
            optimizer.step()  # Perform optimization by adjusting W and b,
            optimizer.zero_grad()  # Clear gradients for next step

        print("accuracy = %s" % net.accuracy(x_test, y_test))
    print("Finished")

# train_gd_model()

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
