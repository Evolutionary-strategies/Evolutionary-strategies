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


def run_master():
    master = Master(nworkers=8)
    while True:
        rewards = master.wait_for_results()
        master.push_results(rewards)

def run_worker():
    worker = Worker()
    noise = SharedNoiseTable()
    params = np.zeros(100)
    seed = np.random()
    while True:
        epsilon = noise.get(seed, 0)
        stdev = 0
        perturbed_params = params + stdev*epsilon
        net = Net(perturbed_params)
        worker.send_result(net.test())
        rewards, seeds = worker.poll_results()


