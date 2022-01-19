import numpy as np
from dist import Master, Worker
from model import Net
from util import *


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
        results = (evaluate_fitnesses(results[0], results[2]), results[1])
        


def silent_worker(lr, noise, sigma, nworkers):
    worker = Worker(-1, lr)
    net = Net()
    params = np.zeros(666560)
    while True:
        results, seeds = worker.poll_run()
        params += calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)     
        print(f"evolution: {calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)}")   
        net.set_params(params)
        reward = net.test()
        print(f"noiseless reward: {reward}")
        worker.send_result(reward, -1)

def run_worker(id, lr, noise, sigma, nworkers):
    worker = Worker(id, lr)
    net = Net()
    params = np.zeros(666560) # øke denne?
    seeds = np.zeros(500)
    while True:
        results, seeds = worker.poll_run()
        params += calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)
        perturbed_params = params + sigma * noise.get(seeds[worker.worker_id], len(params))
        net.set_params(perturbed_params)
        worker.send_result(net.test(), seeds[worker.worker_id])
        


