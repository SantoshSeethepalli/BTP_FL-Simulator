import sys
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from threading import Lock

# Create a print lock for synchronized printing
print_lock = Lock()

def train(model, dataset, client_id=None):
    epochs = 3
    learningRate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = nn.NLLLoss()  # works for any n_classes

    with print_lock:
        print(" ")
        if client_id is not None:
            print(f"Training Client {client_id}")
        else:
            print("Training:")

    for epoch in range(epochs):
        epochLoss = 0
        # Disable tqdm for cleaner output
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

        epochLoss /= len(dataset)
        
        with print_lock:
            print(f"Client {client_id if client_id is not None else ''} Epoch {epoch} Loss: {epochLoss:.4f}")

    return model

def test(model, testSet):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in testSet:
            output = model(input)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    return correct / total
