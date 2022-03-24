from util import load_data
from model import load_model
import foolbox as fb
import numpy as np
import logging
from prettytable import PrettyTable

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
    
    def perform(self, attack, epsilons = [0.03]) -> list[float]:
        accuracy = []
        for eps in epsilons:
            acc = 0
            for i, data in enumerate(testloader, 0):
                images, labels = data
                raw, clipped, is_adv = attack(self.fmodel, images, labels, epsilons=eps)
                acc += 1 - is_adv.float().mean(axis=-1)
            accuracy.append(acc.item()/i)
        
        return accuracy

    def perform_attacks(self, attacks, epsilons = [0.03]) -> dict[list[float]]:
        results = {}
        for attack in attacks:
            results[attack] = self.perform(attack, epsilons)
        logger.info(results)
        return results



def print_data(data, epsilons):
    table = PrettyTable()
    table.field_names = ["Attack"] + epsilons
    for attack in data: 
        table.add_row([attack] + data[attack])
    print(table)

    #fikse fancy grafer



def attack_pipeline(model) -> dict[list[float]]:
    attacks = [
        fb.attacks.LinfFastGradientAttack(), #FGSM
        fb.attacks.L2FastGradientAttack(), #L2 Basic Iterative Method
        fb.attacks.LinfBasicIterativeAttack(), #L-infinity Basic Iterative Method
        fb.attacks.L2ProjectedGradientDescentAttack(), #L2 Projected Gradient Descent
        fb.attacks.LinfProjectedGradientDescentAttack() #L-infinity Projected Gradient Descent
    ]
    epsilons = np.linspace(0.0, 0.2, num=20)

    attack = Attack(model)
    data = attack.perform_attacks(attacks, epsilons)
    print_data(data, epsilons)




# a = Attack()
# print(a.default_accuracy())
# print(a.perform(fb.attacks.LinfFastGradientAttack(), np.linspace(0.0, 0.1, num=10)))
# print(a.perform_attacks([fb.attacks.LinfFastGradientAttack(), fb.attacks.L2FastGradientAttack()]))