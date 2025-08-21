import sys
sys.dont_write_bytecode = True
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import medmnist
from medmnist import INFO
from medmnist.dataset import PathMNIST

from torchvision import transforms

# Choose MedNIST dataset type: 'pathmnist', 'chestmnist', 'dermamnist', etc.
DATA_FLAG = 'pathmnist'  
info = INFO[DATA_FLAG]
n_channels = info['n_channels']  # number of channels
n_classes = len(info['label'])

# Transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5]*n_channels, std=[.5]*n_channels)
])

# Custom dataset wrapper to handle target format
class MedMNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        # Convert target to a scalar by taking first element
        target = target.squeeze()
        return img, target
    
    def __len__(self):
        return len(self.dataset)

def load_dataset(isTrainDataset=True) -> Dataset:
    split = 'train' if isTrainDataset else 'test'
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    medmnistDataset = PathMNIST(
        root=data_dir,
        split=split,
        transform=data_transform,
        download=True
    )
    # Wrap the dataset to handle target format
    return MedMNISTWrapper(medmnistDataset)

def split_client_datasets(dataset, clientNum, roundNum):
    countPerSet = len(dataset) // (clientNum * roundNum)
    clientDatasets = [[] for _ in range(clientNum)]
    for client in range(clientNum):
        for round in range(roundNum):
            low = countPerSet * (round + client * roundNum)
            high = low + countPerSet
            subsetIndices = [i for i in range(low, high)]
            clientDatasets[client].append(Subset(dataset, subsetIndices))
    return clientDatasets

def get_dataloader(dataset, batchSize=64):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchSize, shuffle=True, drop_last=True
    )
    return dataloader