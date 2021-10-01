import torch
import torch.nn as nn
import time
from math import pi

class Lqa(nn.Module):
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """

    def __init__(self, couplings):
        super(Lqa, self).__init__()

        self.couplings = torch.tensor(couplings,dtype=torch.float32)
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1])
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, 1])
        self.weights = torch.zeros([self.n])

    def schedule(self, i, N):
        #annealing schedule
        return i / N

    def energy_ising(self, config):
        # ising energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def energy_full(self, t, g):
        # cost function value
        config = torch.tanh(self.weights)*pi/2
        ez = self.energy_ising(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t*ez*g- (1-t)*ex)


    def minimise(self,
                 step=1,  # learning rate
                 N=200,  # no of iterations
                 g=1.,
                 f=1.):

        self.weights = (2 * torch.rand([self.n]) - 1) * f
        self.weights.requires_grad=True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights],lr=step)

        for i in range(N):
            t = self.schedule(i, N)
            energy = self.energy_full(t,g)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights)
        self.energy = float(self.energy_ising(self.config))

        return self.energy


