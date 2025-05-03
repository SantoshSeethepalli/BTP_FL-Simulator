import torch
from torch.utils.data import DataLoader
from dp_utils import apply_dp

def client_update(model, dataset, epochs=1, dp=False):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            if dp:
                apply_dp(model)
            optimizer.step()
