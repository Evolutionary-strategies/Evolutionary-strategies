from es import *
from util import *
import multiprocessing as mp
import logging
import numpy as np


def launch(nworkers, ismaster):
    mp.log_to_stderr(logging.DEBUG)
    if ismaster:
        master = mp.Process(target = run_master, args = (nworkers,))
        master.start()
    noise = SharedNoiseTable()
    workers = [mp.Process(target=run_worker, args=(x,0.1,noise)) for x in range(0, nworkers-1)]
    workers.append(mp.Process(target=silent_worker, args=(0.1,noise)))
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    if ismaster:
        master.join()


if __name__ == '__main__':    
    launch(5, True)