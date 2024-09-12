### Importing Libraries ###
import numpy as np
import math
import random
import sys
from numpy.linalg import inv
import matplotlib.pyplot as plt

### Import Pyro/Torch Libraries ###
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize


from torch import nn
import os
import logging
from torch.distributions import constraints
smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
torch.set_default_tensor_type(torch.DoubleTensor)

#torch.set_printoptions(precision=8)

def powerIterTorch(cov_mat, num_eigenvalues=10, max_iterations=1000):
    """
    :param cov_mat: The covariance matrix of which we seek the eigenvalues/eigenvectors
    :param num_eigenvalues: Set the number of eigenvalues to compute
    :param max_iterations: Set the maximum number of iterations
    :return: The first num_eigenvalues eigenvalues and eigenvectors are returned.
    """

    # Initialize the list of eigenvalues and eigenvectors
    eigenvalues = []
    eigenvectors = []

    # Iterate to compute the first few eigenvalues and eigenvectors
    for i in range(num_eigenvalues):
        # Initialize the first eigenvector guess
        x = torch.randn(cov_mat.size(0), 1)

        # Power iteration
        for j in range(max_iterations):
            x_new = cov_mat @ x
            x_new_norm = torch.norm(x_new)
            x_new /= x_new_norm
            if torch.norm(x_new - x) < 1e-6:
                break
            x = x_new

        # Compute the eigenvalue corresponding to the eigenvector x
        eigenvalue = (x.t() @ cov_mat @ x) / (x.t() @ x)

        # Store the eigenvector and eigenvalue in the list
        eigenvectors.append(x.flatten())
        eigenvalues.append(eigenvalue.item())

        # Print the eigenvalue and eigenvector
        print("Eigenvalue", i + 1, "=", eigenvalue.item())
        print("Eigenvector", i + 1, "=", x.t().flatten())

        # Deflate the covariance matrix by removing the contribution of the current eigenvector
        cov_mat -= eigenvalue * (x @ x.t())

    return eigenvalues, eigenvectors
def powerIteration(A, etol, max_iter=100):
    # Converting numpy arrays to torch
    if isinstance(A, np.ndarray):
        a = torch.from_numpy(A)
    else:
        a = A

    # Initialization
    ndim = a.size()[0]
    b = torch.ones((ndim, 1))
    b = b + torch.reshape(torch.rand(ndim), (-1, 1))*0.01
    b = b/torch.linalg.vector_norm(b)
    b = b.to(dtype=torch.float64)

    # Main Loop
    counter = 0
    bold = 10*b

    for i in range(0, max_iter):
        bold = b
        b = torch.matmul(a, b)
        b = b/torch.linalg.vector_norm(b)
        counter = counter + 1
        diff = torch.linalg.norm(bold - b)/torch.linalg.norm(bold)
        if diff < etol:
            break
    return b, counter

