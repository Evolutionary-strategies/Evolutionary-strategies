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

def run_master(nworkers):
    master = Master(nworkers)
    results = np.array([np.zeros(nworkers),np.zeros(nworkers)])
    while True:
        print("---------------------")
        print(f"run: {master.run_id}")
        print("---------------------")

        seeds = genseeds(nworkers)
        master.push_run(seeds, results)
        results = master.wait_for_results()

    

def run_worker(id, lr, noise):
    worker = Worker(id, lr)
    net = Net()
    params = np.zeros(243)
    seeds = np.zeros(500)
    while True:
        results, seeds = worker.poll_run()
        params += calc_evolution(results, len(params), noise, worker.learning_rate)
        perturbed_params = params + noise.get(seeds[worker.worker_id], len(params))
        net.set_params(perturbed_params)
        worker.send_result(net.test(), seeds[worker.worker_id])
        


