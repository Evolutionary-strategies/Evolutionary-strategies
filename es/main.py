from nes import *
from util import *
import multiprocessing as mp
import logging
import numpy as np
import os 



log_filename = "logs/accuracy.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode="w", encoding=None, delay=False)
file_handler.setLevel(logging.INFO)


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_filename, encoding='utf-8')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def launch(nworkers, ismaster):
    theta_0 =  np.random.uniform(-1.0, 1.0, 666560)
    mp.log_to_stderr(logging.DEBUG)
    sigma = 0.1
    if ismaster:
        master = mp.Process(target = run_master, args = (nworkers,))
        master.start()
    noise = SharedNoiseTable()
    workers = [mp.Process(target=run_worker, args=(x,0.01,noise, sigma, nworkers, theta_0)) for x in range(0, nworkers)]
    #workers.append(mp.Process(target=silent_worker, args=(0.1,noise, sigma, nworkers)))
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    if ismaster:
        master.join()


if __name__ == '__main__':    
    launch(5, True)