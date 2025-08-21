import sys
sys.dont_write_bytecode = True
from Federated.privacy_techniques.laplace_dp import laplace_mechanism
from Federated.privacy_techniques.gaussian_dp import gaussian_mechanism

def apply_privacy(value, scheme='none', epsilon=1.0, sensitivity=1.0, delta=None):
    """
    Apply the selected privacy scheme to the value.

    Args:
        value (ndarray): Value to be privatized.
        scheme (str): Privacy scheme ('laplace', 'gaussian').
        epsilon (float): Privacy budget.
        sensitivity (float): Sensitivity of the operation.
        delta (float): Delta for Gaussian DP, optional.

    Returns:
        ndarray: Privatized value.
    """
    if scheme == 'laplace':
        return laplace_mechanism(value, epsilon, sensitivity)
    elif scheme == 'gaussian' and delta is not None:
        return gaussian_mechanism(value, epsilon, sensitivity, delta)
    elif scheme == 'none':
        return value
    else:
        print("Seletect between laplace | gaussian | none")
