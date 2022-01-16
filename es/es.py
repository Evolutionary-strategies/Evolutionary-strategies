import numpy as np
from dist import Master, Worker
from model import Net

class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here


    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)

def calc_evolution(rews, seeds, len, noise, lr):
    evo = np.zeros(len)
    for seed, reward in zip(seeds, rews):
        evo += reward * noise.get(seed, len)
    return (lr/len(seeds) * evo)

    

def genseeds(nworkers):
    seeds = np.random.rand(nworkers)
    return seeds

def run_master(nworkers):
    master = Master(nworkers)
    rewards = np.zeros(nworkers)
    while True:
        seeds = genseeds(nworkers)
        master.push_run(seeds, rewards)
        rewards = master.wait_for_results()
        

def run_worker():
    worker = Worker()
    net = Net()
    noise = SharedNoiseTable()
    params = np.zeros(100)
    seeds = np.zeros(2)
    while True:
        prevseeds = seeds
        rewards, seeds = worker.poll_run()
        params += calc_evolution(rewards, prevseeds, len(params), noise, worker.learning_rate)
        perturbed_params = params + noise.get(seeds[worker.worker_id], len(params))
        net.set_params(perturbed_params)
        worker.send_result(net.test())
        


