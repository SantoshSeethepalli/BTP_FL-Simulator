import sys
sys.dont_write_bytecode = True
import torch.nn as nn
import torch.nn.functional as F

# Update input channels for MedNIST
n_channels = 3  # e.g., PathMNIST is RGB
n_classes = 9   # adjust based on your chosen MedNIST dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # Adjust input features for fc1
        # For 28x28 input, conv + pooling reduces to 20*4*4 = 320
        # If input size or channels change, compute accordingly
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*4*4)  # flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
