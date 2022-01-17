from es import run_master, run_worker, SharedNoiseTable
import multiprocessing as mp
import logging
import numpy as np


def launch(nworkers, ismaster):
    mp.log_to_stderr(logging.DEBUG)
    if ismaster:
        master = mp.Process(target = run_master, args = (nworkers,))
        master.start()
    noise = SharedNoiseTable()
    workers = [mp.Process(target=run_worker, args=(x,0.05,noise)) for x in range(0, nworkers)]
    
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    if ismaster:
        master.join()


if __name__ == '__main__':    
    launch(7, True)