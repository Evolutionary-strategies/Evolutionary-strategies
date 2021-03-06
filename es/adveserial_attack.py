
from util import load_data
from model import load_model
import foolbox as fb
import numpy as np
import logging
import multiprocessing as mp
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import json

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

loader = load_data(attack = True)
trainloader = loader[0]

class Attack():
    def __init__(self, model = None) -> None:
        if model is None:
            model = load_model()
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)


    def default_accuracy(self) -> float:
        accuracy = 0
        logger.info("Calculating default accuracy")
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            accuracy += fb.utils.accuracy(self.fmodel, images, labels)
        accuracy /= i
        logger.info("Default accuracy: ", str(accuracy))
        return accuracy
    

    def perform(self, attack, results, epsilons = [0.03], model_name = "None") -> None:
        accuracy = []
        for eps in epsilons:
            acc = 0
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                _, _, is_adv = attack(self.fmodel, images, labels, epsilons=eps)
                acc += 1 - is_adv.float().mean(axis=-1)
            acc = acc.item()/i
            accuracy.append(acc)
            logger.info(model_name + ":" + str(attack) + ": Epsilon: " + str(eps) + ", Accuracy: " + str(acc))
        results[str(attack)] = accuracy


    def perform_attacks(self, attacks, epsilons = [0.03], model_name = "None") -> dict[list[float]]:
        manager = mp.Manager()
        results = manager.dict()
        processes = []
        for attack in attacks:
            logger.info(model_name + ":Performing " + str(attack))
            p = mp.Process(target=self.perform, args=(attack, results, epsilons, model_name))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        logger.info("Results on " + model_name + ": " + str(results))
        return results

attacks = [
    [
        fb.attacks.LinfAdditiveUniformNoiseAttack()
    ]
]
"""
    [
    # fb.attacks.NewtonFoolAttack()
    # fb.attacks.SaltAndPepperNoiseAttack(),
    # fb.attacks.L2CarliniWagnerAttack() 
    # fb.attacks.BoundaryAttack(), #Funker kanskje
    # fb.attacks.EADAttack() #usikker, treg
    ],
    [
        #fb.attacks.L2ContrastReductionAttack(),
        #fb.attacks.L2AdditiveGaussianNoiseAttack() 
        # fb.attacks.LinfFastGradientAttack(),
        # fb.attacks.LinfProjectedGradientDescentAttack(),
        # fb.attacks.LinfAdditiveUniformNoiseAttack(),
        # fb.attacks.LinfDeepFoolAttack(),
        # fb.attacks.LinfBasicIterativeAttack() 
    ],
    [
        # fb.attacks.L2ProjectedGradientDescentAttack(),
        
        #fb.attacks.L2DeepFoolAttack(),        
        #fb.attacks.L2FastGradientAttack(),
        #fb.attacks.L2BasicIterativeAttack()
    ],
"""




epsilons = [
    [
        0.3, 0.8, 1.0
    ]
]

"""Tar inn en dict med modeller, der navnet p?? modellen er n??kkelen og modellen er verdien"""
def model_pipeline(models) -> dict[dict[list[float]]]:
    manager = mp.Manager()
    data = manager.dict()
    processes = []
    for i in range(len(attacks)):
        for model_name in models:
            logger.info(model_name + ":Attacking model")

            model = models[model_name]
            p = mp.Process(target=attack_pipeline, args=(data, model_name, model, attacks[i], epsilons[i], False, False))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    logger.info("Finished attacking")
    logger.info("Data: " + str(data))
    # save_to_json(data._getvalue(), "all_models") Funker ikke, vet ikke hvorfor

    return data


def attack_pipeline(data, model_name, model, attacks, epsilons, table = True, plot = True) -> None:
    attack = Attack(model)
    data[model_name] = attack.perform_attacks(attacks, epsilons, model_name)._getvalue()

    save_to_json(data[model_name], model_name)

    if table:
        print_data(data, epsilons)
    if plot:
        plot_data(data, epsilons)


def print_data(data, epsilons) -> None:
    table = PrettyTable()
    table.field_names = ["Attack"] + list(epsilons)

    for attack in data: 
        row = [attack] + data[attack]
        table.add_row(row)
    logger.info("\n" + str(table))



def plot_data(data, epsilons) -> None:
    logger.info("Plotting data")
    cmap = plt.cm.get_cmap("tab20", len(data))
    for i, attack in enumerate(data):
        y = data[attack]
        plt.plot(epsilons, y, color=cmap(i), label=attack)
    
    plt.xlabel("Epsilon value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over different attacks and epsilons")

    plt.legend()
    plt.savefig("../images/accuracy_plot.png")
    plt.show()


"""data should be in dict format"""
def save_to_json(data, model_name) -> None:
    file_name = "../aa_results/accuracy_data_" + model_name + ".json"
    with open(file_name, 'a') as fp:
        json.dump(data, fp,  indent=4)
