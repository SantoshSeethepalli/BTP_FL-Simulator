from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_datasets():
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return full_dataset, test_dataset

def split_data(dataset, num_clients):
    length = len(dataset) // num_clients
    return random_split(dataset, [length]*num_clients)
