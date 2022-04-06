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
testloader = loader[1]

class Attack():
    def __init__(self, model = None) -> None:
        if model is None:
            model = load_model()
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)


    def default_accuracy(self) -> float:
        accuracy = 0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            accuracy += fb.utils.accuracy(self.fmodel, images, labels)
        return accuracy/i
    

    def perform(self, attack, results, epsilons = [0.03]) -> None:
        accuracy = []
        for eps in epsilons:
            acc = 0
            for i, data in enumerate(testloader, 0):
                images, labels = data
                _, _, is_adv = attack(self.fmodel, images, labels, epsilons=eps)
                acc += 1 - is_adv.float().mean(axis=-1)
            acc = acc.item()/i
            accuracy.append(acc)
            logger.info(str(attack) + ": Epsilon: " + str(eps) + ", Accuracy: " + str(acc))
        results[str(attack)] = accuracy


    def perform_attacks(self, attacks, epsilons = [0.03]) -> dict[list[float]]:
        manager = mp.Manager()
        results = manager.dict()
        processes = []
        for attack in attacks:
            logger.info("Performing " + str(attack))
            p = mp.Process(target=self.perform, args=(attack, results, epsilons))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        logger.info("Results: " + str(results))
        return results



"""Tar inn en dict med modeller, der navnet på modellen er nøkkelen og modellen er verdien"""
def model_pipeline(models) -> dict[dict[list[float]]]:
    data = {}
    for model_name in models:
        logger.info("Attacking model: " + model_name)
        attack_data = attack_pipeline(models[model_name], table=False, plot=False)
        data[model_name] = attack_data._getvalue()

    logger.info("Finished attacking")
    logger.info("Data: " + str(data))

    save_to_json(data)

    return data


def attack_pipeline(model, table = True, plot = True) -> dict[list[float]]:
    attacks = [
        fb.attacks.LinfFastGradientAttack(), #Fast Gradient Sign Method (FGSM)
        fb.attacks.L2ProjectedGradientDescentAttack(),
        fb.attacks.LinfProjectedGradientDescentAttack()
        # fb.attacks.L2AdditiveGaussianNoiseAttack(),
        # fb.attacks.SaltAndPepperNoiseAttack()
        # fb.attacks.BoundaryAttack(),
        # fb.attacks.PointwiseAttack() #funker ikke :(( Blir ikke loada
        #fb.attacks.LinfBasicIterativeAttack(), #L-infinity Basic Iterative Method
        #fb.attacks.L2ProjectedGradientDescentAttack(), #L2 Projected Gradient Descent
        #fb.attacks.LinfProjectedGradientDescentAttack() #L-infinity Projected Gradient Descent
    ]
    epsilons = [0.005, 0.01, 0.3, 0.5]# np.linspace(0.0, 1.0, num=1)

    logger.info("Started to attack model")
    attack = Attack(model)
    data = attack.perform_attacks(attacks, epsilons)

    if table:
        print_data(data, epsilons)
    if plot:
        plot_data(data, epsilons)
    
    return data

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
def save_to_json(data) -> None:
    with open('../accuracy_data.json', 'w') as fp:
        json.dump(data, fp,  indent=4)
