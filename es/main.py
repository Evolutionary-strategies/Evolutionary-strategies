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
    sigma = 0.15                #Kan endres
    learning_rate = 0.01        #Kan endres
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


if __name__ == '__main__':   
    # mod = load_model(path="../models/starting_weights.pt", grad = True)
    # train(mod, 0.8)
    # mod.test(True)
    attack_testing()
    # launch(127, True,True)

