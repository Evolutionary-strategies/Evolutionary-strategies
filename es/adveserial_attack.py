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







def attack_pipeline(model) -> dict[list[float]]:
    attacks = [
        fb.attacks.LinfFastGradientAttack(), #Fast Gradient Sign Method (FGSM)
        fb.attacks.L2FastGradientAttack() #Fast Gradient Method (FGM) funker ikke
        #fb.attacks.LinfBasicIterativeAttack(), #L-infinity Basic Iterative Method
        #fb.attacks.L2ProjectedGradientDescentAttack(), #L2 Projected Gradient Descent
        #fb.attacks.LinfProjectedGradientDescentAttack() #L-infinity Projected Gradient Descent
    ]
    epsilons = np.linspace(0.0, 0.1, num=4)

    logger.info("Started to attack model")
    attack = Attack(model)
    data = {'L2FastGradientAttack(rel_stepsize=1.0, abs_stepsize=None, steps=1, random_start=False)': [0.726890756302521, 0.726890756302521, 0.726890756302521, 0.7270908363345339], 'LinfFastGradientAttack(rel_stepsize=1.0, abs_stepsize=None, steps=1, random_start=False)': [0.726890756302521, 0.4449779911964786, 0.2774109643857543, 0.18517406962785113]}# attack.perform_attacks(attacks, epsilons)
    
    print_data(data, epsilons)
    return data


def print_data(data, epsilons) -> None:
    table = PrettyTable()
    table.field_names = ["Attack"] + list(epsilons)

    for attack in data: 
        row = [attack] + data[attack]
        table.add_row(row)
    logger.info("\n" + str(table))

    plot_data(data, epsilons)



def plot_data(data, epsilons):
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

