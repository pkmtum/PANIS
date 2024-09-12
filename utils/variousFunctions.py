import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def calcRSquared(yTrue, yPred):
    """
    :param yTrue: The true/validation value of the function that we test
    :param yPred: The prediction values for the function that our model/algorithm gives
    :return:
    """
    RSquared = 1 - torch.div(
        torch.sum(torch.linalg.norm(torch.reshape(yTrue - yPred, [yTrue.size(0), -1]), dim=-1) ** 2, dim=0),
        torch.sum(torch.linalg.norm(torch.reshape(yTrue - torch.mean(yTrue, dim=0), [yTrue.size(0), -1]), dim=-1) ** 2,
                  dim=0))
    return RSquared

def calcEpsilon(yTrue, yPred):
    """
    :param yTrue: The true/validation value of the function that we test
    :param yPred: The prediction values for the function that our model/algorithm gives
    :return:
    """
    epsilon = 1/yTrue.size(0) * torch.sum(torch.div(
        torch.linalg.norm(torch.reshape(yTrue - yPred, [yTrue.size(0), -1]), dim=-1),
        torch.linalg.norm(torch.reshape(yTrue, [yTrue.size(0), -1]), dim=-1)), dim=0)
    return epsilon

def makeCGProjection(pde, sampX, sampSolFenics, sampYCoeff):
        pde.sampX
        xTest = sampX
        yTest = sampSolFenics
        pde.shapeFunc.createShapeFuncsConstraint()
        rbfRefSol = pde.shapeFunc.cTrialSolutionParallel(sampYCoeff)
        pde.shapeFunc.createShapeFuncsFree()
        sampYCoeff = pde.shapeFunc.findRbfCoeffs(rbfRefSol)
        YProj = torch.einsum('ij,...j->...i', torch.linalg.inv(pde.shapeFunc.aInvB.t() @ pde.shapeFunc.aInvB) @ pde.shapeFunc.aInvB.t(), sampYCoeff)
        yFProj = torch.einsum('ij,...j->...i', torch.linalg.inv(pde.shapeFunc.aInvBCO.t() @ pde.shapeFunc.aInvBCO) @ pde.shapeFunc.aInvBCO.t(), sampYCoeff)
        yProj = torch.einsum('ij,...j->...i',pde.shapeFunc.aInvB, YProj)
        yProjTotal = torch.einsum('ij,...j->...i',pde.shapeFunc.aInvB, YProj) + torch.einsum('...ij,...j->...i', pde.shapeFunc.aInvBCO, yFProj)
        yProj = pde.shapeFunc.cTrialSolutionParallel(yProj)
        yProjTotal = pde.shapeFunc.cTrialSolutionParallel(yProjTotal)
        return xTest, yTest, yProj, yProjTotal

def setupDevice(cudaIndex, device, dataType):
    np.set_printoptions(formatter={'float': '{: 0.14f}'.format})
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaIndex)

    if device == 'cpu':
        use_cuda=False
        torch.set_default_device('cpu')
    else:
        use_cuda=True
        torch.set_default_device('cuda:0')

    if dataType == 'float':
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print("PyTorch version:", torch.__version__)
    else:
        print("CUDA is not available")

    print(device)
    return device

def createFolderIfNotExists(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
        return
    else:
        return

def memoryOfTensor(tensor):
    num_elements = tensor.numel()

    # Get the size of each element in bytes
    element_size = tensor.element_size()

    # Calculate the total memory used in bytes
    memory_usage_bytes = num_elements * element_size

    # Print the memory usage in MB for easier reading
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    
    return memory_usage_mb
