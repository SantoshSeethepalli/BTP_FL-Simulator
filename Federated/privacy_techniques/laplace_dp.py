import sys
sys.dont_write_bytecode = True
import numpy as np

def laplace_mechanism(value, epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=value.shape)
    return value + noise
