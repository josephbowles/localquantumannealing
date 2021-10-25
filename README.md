Local quantum annealing for Ising optimisation problems
===

This package performs energy minimisation of Ising Hamiltonians as described in arXiv:2108.08064: 
"Quadratic Unconstrained Binary Optimisation via Quantum-Inspired Annealing" 

To cite this code please use doi.org/10.5281/zenodo.5567120

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5567120.svg)](https://doi.org/10.5281/zenodo.5567120)

Dependencies
===

pytorch

Install
===
via pip:

pip install git+https://github.com/josephbowles/localquantumannealing.git

usage
===

A problem instance is specified by a symmetric n x n torch tensor with zero diagonal, encoding the Hamiltonian. 
One then creates a lqa class instance and calls minimise()

Example code:
```
import torch
import localquantumannealing as lqa
```

create random 1000 x 1000 symmetric coupling matrix
```
couplings = torch.rand([1000,1000])
couplings = couplings+couplings.T
couplings = 2*couplings-1
couplings.fill_diagonal_(0.)
```

create a lqa instance
```
machine = lqa.Lqa(couplings)
```

if there is a cuda enabled GPU avaliable, switch to it
```
gpu = torch.device('cuda:0')
machine.to(gpu)
```

run the minimisation with chosen hyperparameters:

```
machine.minimise(step=2, N=1000, g=1, f=0.1)
```

* step: initial step size (fed to Adam gradient descent algorithm)
* N: total number of steps
* g: annealing strength (gamma in the article)
* f: constant that multiplies the random weight initialisation


return the spin configuration, energy and optimisation time

```
config = machine.config
energy = machine.energy
opt_time = machine.opt_time

```

One can also use the class Lqa_basic. This performs simple momentum assisted gradient descent without going through the pytorch
autograd functions. It is thus less memory intensive and suitable for larger problems.
Here, one should specify the momentum when calling minimise. e.g. 

```
machine = lqa.Lqa_basic(couplings)
machine.minimise(step=0.1, N=1000, g=1, f=0.1, mom=0.99)
```
where mom corresponds to the parameter \mu in the article.  

Contact: bowles.physics@gmail.com



