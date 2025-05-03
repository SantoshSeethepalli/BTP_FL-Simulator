import copy
import time
import torch
from torch.utils.data import DataLoader

from model import CNN
from client import client_update
from data_utils import load_datasets, split_data

def evaluate(model, test_data):
    """Evaluate the model on the test dataset and return accuracy."""
    model.eval()
    loader = DataLoader(test_data, batch_size=128)
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    return correct / len(test_data)

def FedAvg(local_models):
    """Federated Averaging: average weights from all local models."""
    avg_model = copy.deepcopy(local_models[0])
    for key in avg_model.state_dict().keys():
        for model in local_models[1:]:
            avg_model.state_dict()[key] += model.state_dict()[key]
        avg_model.state_dict()[key] /= len(local_models)
    return avg_model

def central_training(train_data, test_data, num_epochs):
    """Train a single model centrally for comparison."""
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    epoch_accs = []
    epoch_times = []
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_data)
        elapsed = time.time() - start
        print(f"[CENTRAL] Epoch {epoch}: Acc={acc:.4f}, Time={elapsed:.2f}s")
        epoch_accs.append(acc)
        epoch_times.append(elapsed)
    return epoch_accs, epoch_times

def simulate_federated_learning(num_clients, num_rounds, dp_enabled):
    """Simulate Federated Learning and Centralized Learning for comparison."""
    train_data, test_data = load_datasets()
    client_data = split_data(train_data, num_clients)
    global_model = CNN()

    fl_accs = []
    fl_times = []

    for r in range(1, num_rounds + 1):
        start = time.time()
        local_models = []
        for ds in client_data:
            m = copy.deepcopy(global_model)
            client_update(m, ds, dp=dp_enabled)
            local_models.append(m)

        # Federated Averaging
        global_model = FedAvg(local_models)

        acc = evaluate(global_model, test_data)
        elapsed = time.time() - start
        print(f"[FED] Round {r}: Acc={acc:.4f}, Time={elapsed:.2f}s")
        fl_accs.append(acc)
        fl_times.append(elapsed)

    # Run centralized training for comparison
    cent_accs, cent_times = central_training(train_data, test_data, num_rounds)

    return fl_accs, fl_times, cent_accs, cent_times
