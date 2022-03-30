from util import load_data
from model import load_model
import foolbox as fb
import numpy as np
import logging
import torch
import torchvision.transforms as T
from PIL import Image

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

loader = load_data(attack = True)
trainloader = loader[0]
testloader = loader[1]
classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Attack():
    def __init__(self, model = None) -> None:
        if model is None:
            model = load_model()
        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

    def adv_bool(self, attack, data, epsilons = [0.03]) -> list[bool]:
        image, label = data
        _, _, is_adv = attack(self.fmodel, image, label, epsilons=epsilons)
        return is_adv[0].tolist()

    


"""Skal returnere True når bildet kan brukes"""
def usable_img(list1, list2) -> bool:
    if len(list1) != len(list2):
        return False

    if list1[0] == False or list2[0] == False: # hmm, muligens feil, se på det når man skal fikse skikkelige eksempler
        return False

    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return True
    
    return False

def print_img(attack, model, img, label, eps, i, model_nr) -> None:
    _, clipped1, _ = attack(model, img, label, epsilons=eps)
    transform = T.ToPILImage()
    img1 = transform(clipped1[0])
    name1 = "../images/" + str(classes[label[0].item()]) + "_model" + model_nr + "_i" + str(i) + ".png"
    img1.save(name1)



def print_image(model1, model2, attack, epsilons = [0.03]) -> None:
    logger.info("Finding images")
    attack1 = Attack(model1)
    attack2 = Attack(model2)

    for data in testloader:
        is_adv1 = attack1.adv_bool(attack, data, epsilons)
        is_adv2 = attack2.adv_bool(attack, data, epsilons)
        if usable_img(is_adv1, is_adv2):
            logger.info("Found usable image")
            logger.info("is_adv1:" + str(is_adv1))
            logger.info("is_adv2:" + str(is_adv2))

            org_img, label = data

            for i in range(len(is_adv1)):
                print_img(attack, attack1.fmodel, org_img, label, epsilons[i], i, "1")
                print_img(attack, attack2.fmodel, org_img, label, epsilons[i], i, "2")
            break






net1 = load_model()
net2 = load_model("../models/starting_weights.pt")
print_image(net1, net2, fb.attacks.LinfFastGradientAttack(), epsilons = np.linspace(0.0, 0.1, num=4))

