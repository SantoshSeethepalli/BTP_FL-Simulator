import torch
from torch.utils.data import DataLoader
from dp_utils import apply_dp, PrivacyScheme

def client_update(model, dataset, epochs=1, dp=False, dp_scheme=PrivacyScheme.GAUSSIAN, 
                 epsilon=1.0, delta=1e-5):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset_size = len(dataset)

    for _ in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            
            if dp:
                apply_dp(model, epsilon, delta, dataset_size, dp_scheme)
            
            optimizer.step()
