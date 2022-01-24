import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def evaluate_fitnesses(rewards, noiseless_reward):
    rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    return softmax(rewards)
    #return rewards

class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here


    def get(self, i, dim):
        return self.noise[i:i + dim]


def calc_evolution(results, length, noise, lr, sigma, nworkers):
    evo = np.zeros(length)
    rews = results[0]
    seeds = results[1]
    for seed, reward in zip(seeds, rews):
        evo += reward * noise.get(int(seed), length)
    return (lr/(sigma*nworkers) * evo)


def genseeds(nworkers):
    seeds = np.random.randint(0, 240000000, nworkers)
    return seeds

def load_data():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainloader, testloader)