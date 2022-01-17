import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
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
            self.conv1[0].weigth = nn.Parameter(params[0]) # params[0] må være en tensor
            self.conv2[0].weigth = nn.Parameter(params[1]) # params[1] må være en tensor
        
    # kilde: https://discuss.pytorch.org/t/how-to-output-weight/2796
    # Printer, men vet ikke om det er "riktige" tensorer den printer
    def print_layers(self):
        for param in self.parameters():
            print(param.data)

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

net = Net()
#net.set_params([0,0])

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

net.test()