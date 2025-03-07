import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, 
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, 
                              transform=transform, download=True)

x_train = train_dataset.data.float() / 255
y_train = train_dataset.targets

x_test = test_dataset.data.float() / 255
y_test = test_dataset.targets


W_1 = np.random.rand(784,10) - 0.5
b_1 = np.random.rand(10, 1) - 0.5 
W_2 = np.random.rand(10, 10) - 0.5
b_2 = np.random.rand(10, 1) - 0.5
A_layers = []
for i in range(len(x_train)):
   A_layers.append(x_train[i].reshape(784, 1))
