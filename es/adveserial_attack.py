
from util import load_data
from model import load_model
import foolbox as fb

loader = load_data()
trainloader = loader[0]
testloader = loader[1]

model = load_model()
fmodel = fb.PyTorchModel(model, bounds=(0,1))
attack = fb.attacks.LinfFastGradientAttack()



accuracy = 0
robust_accuracy = 0
i = 0
for data in testloader:
    images, labels = data
    accuracy += fb.utils.accuracy(fmodel, images, labels)

    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
    robust_accuracy += 1 - is_adv.float().mean(axis=-1)

    i += 1

print(accuracy/i)
print(robust_accuracy.item()/i)
