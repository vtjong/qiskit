import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms

class Data():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.trainloader = self.trainingdata()
        self.testloader =  self.testingdata()

    def trainingdata(self):
        # Concentrating on the first 100 samples
        n_samples = 100

        X_train = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

        # Leaving only labels 0 and 1 
        idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                    np.where(X_train.targets == 1)[0][:n_samples])

        X_train.data = X_train.data[idx]
        X_train.targets = X_train.data[idx]
        # X_train.targets = X_train.targets[idx]

        return torch.utils.data.DataLoader(X_train, batch_size=self.batch_size, shuffle=True)

    def testingdata(self):
        n_samples = 50

        X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

        idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                    np.where(X_test.targets == 1)[0][:n_samples])

        X_test.data = X_test.data[idx]
        X_test.targets = X_test.targets[idx]

        return torch.utils.data.DataLoader(X_test, batch_size=self.batch_size, shuffle=True)
