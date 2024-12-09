import torch.nn as nn
import torch


class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Fully connected layer
        # CIFAR-10 images are 32x32. After two pooling layers:
        # - Size is reduced to 32 -> 16 -> 8
        # - With 32 output channels from conv2, input size is 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(self.relu1(self.conv1(x)))

        # Second convolutional layer with ReLU and pooling
        x = self.pool(self.relu2(self.conv2(x)))

        # Flattening the tensor
        x = x.view(x.size(0), -1)  # Batch size remains unchanged

        # Fully connected layer
        x = self.fc1(x)
        return x
