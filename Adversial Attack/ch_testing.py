import sys
sys.path.append('/home/Code/Bachelor/Evolutionary-strategies/es' )
from model import Net
"""
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

"""
for p in sys.path:
    print( p )


def load_model(name = "example"):
    print("Loading model")
