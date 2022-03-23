from es import *
from util import *
from model import *
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
logging.basicConfig(filename=log_filename)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def launch(nworkers, ismaster, loadparams=False):
    if loadparams:
        net = load_model("../models/nes_model1.pt")
        theta_0 = np.concatenate((
            net.conv1.weight.detach().numpy().flatten(),
            net.conv1.bias.detach().numpy().flatten(),
            net.conv2.weight.detach().numpy().flatten(), 
            net.conv2.bias.detach().numpy().flatten(), 
            net.fc1.weight.detach().numpy().flatten(), 
            net.fc1.bias.detach().numpy().flatten(), 
            net.fc2.weight.detach().numpy().flatten(),         
            net.fc2.bias.detach().numpy().flatten()
            ))

    else:
        theta_0 =  np.random.uniform(-1.0, 1.0, 666890) #666890
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

def gd_testing():
    net = Net(True)
    train(net, acc_limit=0.5)

if __name__ == '__main__':    
    launch(127, True,True)
    # gd_testing()
