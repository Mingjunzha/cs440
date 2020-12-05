# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output




        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.cnn = nn.Sequential(nn.Conv2d(3,9, kernel_size=4), nn.ReLU())
        self.cnnTrans = nn.Sequential(nn.ConvTranspose2d(9,3, kernel_size=4), nn.ReLU())
        self.nerualNetwork = nn.Sequential(nn.Linear(in_size, 128),nn.ReLU(),nn.Linear(128, 128),nn.ReLU(),nn.Linear(128,out_size))
        self.lrate = lrate
        self.optims = optim.Adagrad(self.parameters(), self.lrate,weight_decay=0.004)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = x.view(-1,3,32,32)
        x =self.cnnTrans(self.cnn(x)).view(-1,3*32*32)
        return self.nerualNetwork(x)
    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optims.zero_grad()
        lossFunction = self.loss_fn(self.forward(x), y)
        lossFunction.backward()
        self.optims.step()

        return lossFunction.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    
    loss_fn = nn.CrossEntropyLoss()
    lrate = 0.0037
    net = NeuralNet(lrate, loss_fn,len(train_set[0]), 2)
    losses = []
    for i in range(n_iter):
        nets = torch.randperm(len(train_set))
        labels = train_labels[nets[:batch_size]]
        trains = train_set[nets[:batch_size]]
        steps = net.step(trains, labels)
        losses.append(steps)

    network = net.forward(dev_set).detach().numpy()
    yhats = np.argmax(network,axis=1)
    return losses, yhats, net