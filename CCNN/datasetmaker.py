import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Data():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.n_train = 400    # Size of the train dataset
        self.n_test = 10     # Size of the test dataset

        self.trainloader = self.trainingdata()
        self.testloader =  self.testingdata()
        self.SAVE_PATH = "data/" # Data saving folder

    def trainingdata(self):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=self.batch_size, shuffle=True)

        return train_loader

    def testingdata(self):

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='./data',
            train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=self.batch_size, shuffle=False)
    
        return test_loader
