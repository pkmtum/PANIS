import numpy as np
import torch
from scipy import integrate

def trapzInt2D(f_xy):
    out = torch.trapezoid(f_xy, torch.linspace(0, 1, f_xy.size(dim=-1)), dim=-1)
    return torch.trapezoid(out, torch.linspace(0, 1, f_xy.size(dim=-2)), dim=-1)

def trapzInt2DParallel(f_xy):
    out = torch.trapezoid(f_xy, torch.linspace(0, 1, f_xy.size(dim=-1)), dim=-1)
    return torch.trapezoid(out, torch.linspace(0, 1, f_xy.size(dim=-2)), dim=-1)

def simpsonInt2DParallel(f_xy):
    h2 = 1 / (f_xy.size(dim=-1) - 1)
    h1 = 1 / (f_xy.size(dim=-2) - 1)
    f_xy = (torch.sum(f_xy[..., 1:-1:2], dim=-1) * 4. +
                              torch.sum(f_xy[..., 2:-1:2], dim=-1) * 2. + (f_xy[...,  0] + f_xy[..., -1])) * h2 / 3.
    f_xy = (torch.sum(f_xy[..., 1:-1:2], dim=-1) * 4. +
                              torch.sum(f_xy[..., 2:-1:2], dim=-1) * 2. + (f_xy[...,  0] + f_xy[..., -1])) * h1 / 3.
    return f_xy

def booleInt2DParallel(f_xy): # It is Wrong, It doesn't work for now!
    h2 = 1 / (f_xy.size(dim=2) - 1)
    h1 = 1 / (f_xy.size(dim=1) - 1)
    f_xy = (torch.sum(f_xy[:, :, 1:-1:4], dim=2) * 7. +
            torch.sum(f_xy[:, :, 2:-1:4], dim=2) * 32. +
            torch.sum(f_xy[:, :, 3:-1:4], dim=2) * 12. +
            torch.sum(f_xy[:, :, 4:-1:4], dim=2) * 32. +
            (f_xy[:, :, 0] + f_xy[:, :, -1])) * h2 * 2. / 45.

    f_xy = (torch.sum(f_xy[:, 1:-1:4], dim=1) * 7. +
            torch.sum(f_xy[:, 2:-1:4], dim=1) * 32. +
            torch.sum(f_xy[:, 3:-1:4], dim=1) * 12. +
            torch.sum(f_xy[:, 4:-1:4], dim=1) * 32. +
            (f_xy[:, 0] + f_xy[:, -1])) * h1 * 2. / 45.
    return f_xy


def testIntegrationMethods():
    analRes = 1.7182818284590452353602874713526624977572470936999595749669676277
    x = torch.linspace(0, 1, 51)
    X, Y = torch.meshgrid(x, x)

    # Define the function f(x, y)
    def f(x, y):
        return x * torch.exp(x + y)

    # Evaluate the function over the grid
    Z = f(X, Y).repeat(1, 1, 1)
    trapzTest = trapzInt2DParallel(Z).detach().cpu().numpy()
    simpsSciTest = integrate.simps(integrate.simps(Z.cpu().detach().numpy(), np.linspace(0, 1, 51)), np.linspace(0, 1, 51))
    simpsMineTest = simpsonInt2DParallel(Z).detach().cpu().numpy()
    #booleMineTest = booleInt2DParallel(Z).detach().cpu().numpy()
    comparisonSimMineAnal = np.abs(analRes - simpsMineTest)
    compSimSciAnal = np.abs(analRes - simpsSciTest)
    compTrapzAnal = np.abs(trapzTest - analRes)
    #compBoolMineAnal = np.abs(booleMineTest - analRes)
    tess = 'tess'
    return