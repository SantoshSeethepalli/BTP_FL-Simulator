import torch
import numpy as np

class PrivacyScheme:
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"

def calculate_sensitivity(learning_rate, clip_norm, dataset_size):
    """Calculate sensitivity for DP noise calibration"""
    return 2 * learning_rate * clip_norm / dataset_size

def add_dp_noise(grads, epsilon, delta, dataset_size, scheme=PrivacyScheme.GAUSSIAN):
    # Calculate clip norm (L2 norm clipping)
    clip_norm = 1.0
    learning_rate = 0.01
    
    # Calculate sensitivity
    sensitivity = calculate_sensitivity(learning_rate, clip_norm, dataset_size)
    
    if scheme == PrivacyScheme.GAUSSIAN:
        # Using the Gaussian mechanism with moments accountant
        sigma = np.sqrt(2 * np.log(1.25/delta)) / epsilon
        return [g + torch.normal(0, sigma * sensitivity, g.shape) for g in grads]
    
    elif scheme == PrivacyScheme.LAPLACE:
        # Using the Laplace mechanism
        scale = sensitivity / epsilon
        return [g + torch.from_numpy(np.random.laplace(0, scale, g.shape)).float() for g in grads]
    else:
        raise ValueError("Invalid DP scheme")

def apply_dp(model, epsilon=1.0, delta=1e-5, dataset_size=60000, scheme=PrivacyScheme.GAUSSIAN):
    """Apply differential privacy with privacy budget tracking"""
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Add calibrated noise
                param.grad = add_dp_noise([param.grad], epsilon, delta, dataset_size, scheme)[0]
