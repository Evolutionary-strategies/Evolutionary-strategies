import numpy as np
from dist import Master, Worker
from model import Net
from util import *
import logging



logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_master(nworkers):
    master = Master(nworkers)
    results = np.array([np.zeros(nworkers),np.zeros(nworkers)])
    while True:
        seeds = genseeds(nworkers)
        master.push_run(seeds, results)
        results = master.wait_for_results()
        results = (evaluate_fitnesses(results[0], results[2]), results[1])
        


"""def silent_worker(lr, noise, sigma, nworkers, theta_0):
    worker = Worker(-1, lr)
    net = Net()
    params = theta_0
    while True:
        results, seeds = worker.poll_run()
        params += calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)     
        print(f"evolution: {calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)}")   
        net.set_params(params)
        reward = net.test()
        if worker.run_id % 100 == 1:
            net.save_model("nes_model.pt")
        print(f"noiseless reward: {reward}")
        worker.send_result(reward, -1)"""

def run_worker(id, lr, noise, sigma, nworkers, theta_0):
    worker = Worker(id, lr)
    net = Net()
    params = theta_0
    seeds = np.zeros(500)
    while True:
        results, seeds = worker.poll_run()
        params += calc_evolution(results, len(params), noise, worker.learning_rate, sigma, nworkers)
        """worker.run_id % 100 == 1 and"""
        if  worker.run_id % 100 == 1 and worker.worker_id == 1:
            logger.info(f"run: {worker.run_id}")
            accuracy = net.test(log=True)
            worker.send_result(accuracy, seeds[worker.worker_id])
            if accuracy > 0.73 and accuracy < 0.74:
                net.save_model("es_model_sigma015_acc073_2.pt")
            
        else:
            perturbed_params = params + sigma * noise.get(seeds[worker.worker_id], len(params))
            net.set_params(perturbed_params)
            worker.send_result(net.test(), seeds[worker.worker_id])
        


