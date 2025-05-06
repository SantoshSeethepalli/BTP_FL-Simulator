import sys
sys.dont_write_bytecode = True
import numpy as np

def gaussian_mechanism(value, epsilon, sensitivity, delta):
    scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, scale, size=value.shape)
    return value + noise
