#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
from math import pi


class Lqa_basic():
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """
    def __init__(self, couplings):
        super(Lqa_basic, self).__init__()
        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1])
        self.weights = (2 * torch.rand(self.n) - 1) * 0.1
        self.velocity = torch.zeros(self.n)
        self.grad = torch.zeros(self.n)

    def forward(self, t, step, mom, g):
        # Implements momentum assisted gradient descent update
        w = torch.tanh(self.weights)
        a = 1 - torch.tanh(self.weights) ** 2
        # spin x,z values
        z = torch.sin(w * pi / 2)
        x = torch.cos(w * pi / 2)

        # gradient
        self.grad = ((1 - t) * z + 2 * t * g * torch.matmul(self.couplings, z) * x) * a * pi / 2
        # weight update
        self.velocity = mom * self.velocity - step * self.grad
        self.weights = self.weights + self.velocity

    def schedule(self, i, N):
        return i / N

    def energy_ising(self, config):
        # energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def minimise(self,
                 step=2,  # step size
                 g=1,  # gamma in the article
                 N=200,  # no of iterations
                 mom=0.99,  # momentum
                 f=0.1  # multiplies the weight initialisation
                 ):

        self.weights = (2 * torch.rand(self.n) - 1) * f
        self.velocity = torch.zeros([self.n])
        self.grad = torch.zeros([self.n])

        time0 = time.time()

        for i in range(N):
            t = self.schedule(i, N)
            self.forward(t, step, mom, g)

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights.detach())
        self.energy = float(self.energy_ising(self.config))

        print('min energy ' + str(self.energy))
