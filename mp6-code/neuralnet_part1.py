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
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.nerualNetwork = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, out_size))
        self.lrate = lrate
        self.optims = optim.SGD(self.nerualNetwork.parameters(), self.lrate,weight_decay = 0.004)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.nerualNetwork(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        
        lossFunction = self.loss_fn(self.forward(x), y)
        lossFunction.backward()

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
    lrate = 0.03
    #print(len(train_set[0]))
    net = NeuralNet(lrate, loss_fn, len(train_set[0]), 2)
    losses = []
    mean = train_set.mean()
    std = train_set.std()
    train = (train_set-mean)/std
    for i in range(n_iter):
        size = len(train_set)//100
        if(i >=size):
            trains = train[(i-size-2)*batch_size:(i-size-1)*batch_size]
            label = train_labels[(i-size-2)*batch_size:(i-size-1)*batch_size]
        else:
            label = train_labels[i*batch_size:(i+1)*batch_size]
            trains = train[i*batch_size:(i+1)*batch_size]
        net.optims.zero_grad()
        steps = net.step(trains, label)
        net.optims.step()
        losses.append(steps)
    
    dev = (dev_set-mean)/std
    network = net(dev).detach().numpy()
    yhats = np.argmax(network,axis=1)
    return losses, yhats, net