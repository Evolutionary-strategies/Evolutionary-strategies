from util import load_data
from model import load_model
import foolbox as fb
import numpy as np
import logging
import multiprocessing as mp
from prettytable import PrettyTable
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

loader = load_data()
trainloader = loader[0]
testloader = loader[1]

# Kjører kun på en CPU
class Attack():
    def __init__(self, model = None) -> None:
        if model is None:
            model = load_model()
        self.fmodel = fb.PyTorchModel(model, bounds=(0,1))


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
                raw, clipped, is_adv = attack(self.fmodel, images, labels, epsilons=eps)
                acc += 1 - is_adv.float().mean(axis=-1)
            acc = acc.item()/i
            accuracy.append(acc)
            logger.info(str(attack) + ": Accuracy at " + str(eps) + " epsilon: " + str(acc))
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



def print_data(data, epsilons) -> None:
    table = PrettyTable()
    table.field_names = ["Attack"] + list(epsilons)

    for attack in data: 
        row = [attack] + data[attack]
        table.add_row(row)
    logger.info("\n" + str(table))
    
    #TODO: fikse fancy grafer


def attack_pipeline(model) -> dict[list[float]]:
    attacks = [
        fb.attacks.LinfFastGradientAttack(), #FGSM
        fb.attacks.L2FastGradientAttack() #L2 Basic Iterative Method
        #fb.attacks.LinfBasicIterativeAttack(), #L-infinity Basic Iterative Method
        #fb.attacks.L2ProjectedGradientDescentAttack(), #L2 Projected Gradient Descent
        #fb.attacks.LinfProjectedGradientDescentAttack() #L-infinity Projected Gradient Descent
    ]
    epsilons = np.linspace(0.0, 0.1, num=1)

    logger.info("Started to attack model")
    attack = Attack(model)
    data = attack.perform_attacks(attacks, epsilons)
    
    print_data(data, epsilons)
    return data


# Ikke testa
def plot_data(data, epsilons):
    logger.info("Plotting data")
    for attack in data:
        print(attack)
        y = data[attack]
        plt.plot(epsilons, y, color='r', label=attack)
    
    plt.xlabel("Epsilon value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over different attacks and epsilons")

    plt.legend()
    plt.show()