"""from email.mime import image
from xmlrpc.client import Boolean
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


    Used to create image examples, and not necessary for attacks
    def create_image_list(self, attack, epsilons = [0.03]) -> list[list[bool]]:
        logger.info("Creating image_list")
        image_list = []
        for i, data in enumerate(testloader):
            image = []
            for eps in epsilons:
                images, labels = data
                _, _, is_adv = attack(self.fmodel, images, labels, epsilons=eps)
                image.append(is_adv)
            image_list.append(image)
            if i>10:
                break
        return image_list






def attack_pipeline(model) -> dict[list[float]]:
    attacks = [
        fb.attacks.LinfFastGradientAttack(), #Fast Gradient Sign Method (FGSM)
        fb.attacks.L2FastGradientAttack() #Fast Gradient Method (FGM)
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
    


"""Skal returnere True når bildet kan brukes"""
def usable_img(list1, list2) -> bool:
    if len(list1) != len(list2):
        return False
    
    if list1[0] == False or list2[0] == False:
        return False

    for i in range(len(list1)):
        print(list1[i], ":", list2[i])
        if list1[i].tolist() != list2[i].tolist():
            return True
    
    return False

def find_img(model1, model2, attack, epsilons = [0.03]) -> list[list[bool]]:
    logger.info("Finding images")
    attack1 = Attack(model1)
    attack2 = Attack(model2)
    image_list1 = attack1.create_image_list(attack, epsilons)
    image_list2 = attack2.create_image_list(attack, epsilons)
    logger.info("Created image_list")
    print("len(image_list1):")
    print(len(image_list1))
    print(type(image_list1))

    if len(image_list1) != len(image_list2):
        return # Throw error?

    # må fikses så jeg er sikker på at den faktisk har funnet et bilde!
    selected_img_index = 0
    for i in range(len(image_list1)):
        print("type(image_list1[i])")
        print(type(image_list1[i]))
        if usable_img(image_list1[i], image_list2[i]):
            print(i)
            selected_img_index = i
            break
    
    logger.info("Selected specific image")
    org_img, label = testloader[selected_img_index] # er ikke indexbar. må finne dette ut! Eventuelt loope
    images1 = []
    images2 = []
    for i in len(image_list1[selected_img_index]):
        eps = epsilons[i]
        _, clipped1, _ = attack(model1, org_img, label, epsilons=eps)
        print(clipped1)
        print(type(clipped1))
        images1.append(clipped1)
        _, clipped2, _ = attack(model2, org_img, label, epsilons=eps)
        images2.append(clipped2)
    
    
net1 = load_model()
net2 = load_model("../models/starting_weights.pt")
find_img(net1, net2, fb.attacks.L2FastGradientAttack(), epsilons = np.linspace(0.0, 0.1, num=4))


    




"""