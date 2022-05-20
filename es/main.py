from es import *
from util import *
from model import *
from adveserial_attack import *
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
        net = load_model("../models/starting_weights.pt")
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
    sigma = 0.15                #Changable values
    learning_rate = 0.01        #Changable values
    if ismaster:
        master = mp.Process(target = run_master, args = (nworkers,))
        master.start()
    noise = SharedNoiseTable()
    workers = [mp.Process(target=run_worker, args=(x, learning_rate, noise, sigma, nworkers, theta_0)) for x in range(0, nworkers)]
    #workers.append(mp.Process(target=silent_worker, args=(0.1,noise, sigma, nworkers)))
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    if ismaster:
        master.join()

def attack_testing():
    models = {
        "gd_model054": load_model(path="../models/gd_model054.pt"),
        "nes_model_sigma01": load_model(path="../models/nes_model_sigma01.pt"),
        "nes_model_sigma015": load_model(path="../models/nes_model_sigma015.pt"),
        "es_model_sigma015_acc073_1": load_model(path="../models/es_model_sigma015_acc073_1.pt"),
	    "es_model_sigma015_acc073_2": load_model(path="../models/es_model_sigma015_acc073_2.pt")
    }
    model_pipeline(models)

def train_gd(limit = 1):
    net = load_model("../models/starting_weights.pt", True)
    train(net, limit)

if __name__ == '__main__':   
    """ Training should be done on the training branch. """
    """ To launch training of NES run following code: """
    # launch(3, True,True)
    
    """ Change attacks and perturbation budget within the adversarial_attack.py file"""
    """ To test models, change path names and run following code: """
    # attack_testing()

    """ To train GD-model, change the desired accuracy and run following code: """
    # train_gd(0.5)
