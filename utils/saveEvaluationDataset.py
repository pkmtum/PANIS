import torch

def saveDataset(cond, sol, createNewCondField, xCoeff, yCoeff, saveEvalDataset=False, path=None):
    xFNO_data = cond.cpu()
    yFNO_data = sol.cpu()
    FNO_datasetDict = sample_dict = {'x': xFNO_data,
                'y': yFNO_data}
    torch.save(FNO_datasetDict, path)

def importDataset(datapath, device):
    a = torch.load(datapath)['x'].double().to(device)
    u = torch.load(datapath)['y'].double().to(device)
    return a, u

def saveDatasetAll(cond, solFenics, xCoeff, yCoeff, sol, eigVals, eigVecs, createNewCondField, saveEvalDataset=False, path=None):
    xFNO_data = cond.cpu()
    yFNO_data = solFenics.cpu()
    yrbf = sol.cpu()
    xCoeff_data = xCoeff.cpu()
    yCoeff_data = yCoeff.cpu()
    eigVals = eigVals.cpu()
    eigVecs = eigVecs.cpu()
    FNO_datasetDict = sample_dict = {'x': xFNO_data,
                'y': yFNO_data, 'xCoeff': xCoeff_data, 'yCoeff': yCoeff_data, 'yRbf': yrbf, 'eigVals': eigVals, 'eigVecs': eigVecs}
    torch.save(FNO_datasetDict, path)

def importDatasetAll(datapath, device):
    a = torch.load(datapath)['x'].double().to(device)
    u = torch.load(datapath)['y'].double().to(device)
    uRbf = torch.load(datapath)['yRbf'].double().to(device)
    xCoeff = torch.load(datapath)['xCoeff'].double().to(device)
    yCoeff = torch.load(datapath)['yCoeff'].double().to(device)
    eigVals = torch.load(datapath)['eigVals'].double().to(device)
    eigVecs = torch.load(datapath)['eigVecs'].double().to(device)
    return a, u, xCoeff, yCoeff, uRbf, eigVals, eigVecs