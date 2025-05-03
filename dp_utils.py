import torch

def add_dp_noise(grads, noise_scale):
    return [g + torch.randn_like(g) * noise_scale for g in grads]

def apply_dp(model, noise_scale=0.1):
    with torch.no_grad():
        for param in model.parameters():
            param.grad = add_dp_noise([param.grad], noise_scale)[0]
