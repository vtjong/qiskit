from datetime import date
from pampy import match

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import adabound
import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt

from CCNN.net import Net
from datasetmaker import Data

###### SWEEPS ########
config_defaults = {
    "epochs": 20,
    "batch_size": 1,
    "learn_rate": 0.001,
    "optimizer": "adam",
    "momentum": 0.9,
    "dropout": 0.2,
}

wandb.require("service")
wandb.setup()
wandb.init(config=config_defaults)
config = wandb.config

###### DEVICE CONFIG ########
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=10)
cuda_server = 2
device = torch.device(cuda_server if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

######## DATA SETS ##########
batch_size = config.batch_size
today = date.today()
today = today.strftime("%b-%d-%Y")


class Train(object):
    def __init__(self, trainloader, testloader, model):

        self.trainloader = trainloader
        self.testloader = testloader

        # Initialize digital twin model
        self.model = model

    def train(self):
        # Choose an optimizer
        optimizer = match(
            config.optimizer,
            "sgd",
            optim.SGD(
                self.model.parameters(), lr=config.learn_rate, momentum=config.momentum
            ),
            "adam",
            optim.Adam(
                self.model.parameters(), lr=config.learn_rate, weight_decay=1e-6
            ),
            "adabound",
            adabound.AdaBound(
                self.model.parameters(),
                lr=config.learn_rate,
                final_lr=config.learn_rate,
            ),
        )
        # error = nn.NLLLoss()
        error = nn.NLLLoss()
        self.model.train()
        loss_list=[]
        batch_idx = 0

        # Enter training loop
        for epoch in range(config.epochs):
            total_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = error(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())
                # wandb
                wandb.log({"batch loss": loss.item()})

            loss_list.append(sum(total_loss)/len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
                100. * (epoch + 1) / config.epochs, loss_list[-1]))

        np.savetxt(today + "hybrid_loss.csv", np.asarray(loss_list))

    def test(self):
        model = self.model
        error = nn.NLLLoss()
        self.model.eval()
        # Run the model on some test examples
        correct = 0
        total_loss = []
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.testloader):
                # Forward pass
                outputs = model(inputs)
                outputs_plt = outputs.flatten().numpy()
                actual_plt = labels.flatten().numpy()

                # Evaluate
                pred = outputs.argmax(dim=1,keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
                # Store
                predictions.append(outputs_plt)
                actuals.append(actual_plt)

                # regular loss calculations
                loss = error(outputs, labels)
                total_loss.append(loss.item())

                # wandb
                wandb.log({"test loss": loss.data.item()})

            print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(sum(total_loss) / len(total_loss), correct / len(self.testloader) * 100)) 

        # Each new line represents a different test batch
        # Each line has batch_size *
        # Save
        np.savetxt(today + "vers2_actuals_nonlinear.csv", actuals)
        np.savetxt(today + "vers2_predictions_nonlinear.csv", predictions)

def main():
    data = Data(batch_size)
    trainloader = data.trainloader
    testloader = data.testloader
    model_net = Net()

    session = Train(
        trainloader, testloader, model=model_net)
    session.train()
    session.test()


if __name__ == "__main__":
    main()
