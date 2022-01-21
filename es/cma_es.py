import numpy as np
import math
def cma_es():
    N = 666560
    mean = np.random.uniform(-1.0, 1.0, 666560)
    sigma = 0.3

    lamb = 32
    mu = lamb/2
    w = math.log(mu+1/2) - math.log(np.array([x for x in range(1,mu)]))
    print(w)
    return 0


cma_es()