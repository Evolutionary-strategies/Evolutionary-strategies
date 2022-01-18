import numpy as np

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def evaluate_fitnesses(rewards, noiseless_reward):
    rewards = [(x - noiseless_reward) * 100 for x in rewards]
    return softmax(rewards)

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


def calc_evolution(results, length, noise, lr):
    evo = np.zeros(length)
    rews = results[0]
    seeds = results[1]
    for seed, reward in zip(seeds, rews):
        evo += reward * noise.get(int(seed), length)
    return (lr/length * evo)


def genseeds(nworkers):
    seeds = np.random.randint(0, 240000000, nworkers)
    return seeds