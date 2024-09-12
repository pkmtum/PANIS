### Importing Libraries ###
import numpy as np
import math
import random
import pandas as pd
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
import time
#import fenics as df
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from textwrap import wrap
from model.pde.pdeForm2D import pdeForm
from model.pde.shapeFunctions.hatFunctions import rbfInterpolation
from utils.variousFunctions import calcRSquared, calcEpsilon
import shutil
from scipy.ndimage import zoom
import torch


class postProcessing:
    def __init__(self, path='./results/data/', fpath='./results/figs/', displayPlots=True):
        self.path = path
        self.fpath = fpath
        if not os.path.exists(fpath):
            # If it doesn't exist, create the folder
            os.makedirs(fpath)
        if not os.path.exists(path):
            # If it doesn't exist, create the folder
            os.makedirs(path)
        self.defaultReadingList = [
                ['intGrid', 1],
                ['rbfGrid', 1],
                ['xPred',  1],
                ['solCoeffMean', 1],
                ['solPred',  1],
                ['psiEvolution', 1],
                ['phiEvolution', 1],
                ['residualEvolution', 1],
                ['elboEvolution', 1],
                ['movAvgElbo', 1],
                ['movAvgRes', 1],
                ['elboMinMaxEvolution', 1],
                ['relativeImprovementOfPhi', 1],
                ['grads', 1],
                ['gradsNorm', 1],
                ['phiGradEvolution', 1],
                ['psiGradEvolution', 1],
                ['jointGradEvolution', 1],
                ['sigmaEvolution', 1],
                ['solUpLowPred', 1],
                ['solSamplesMean', 1],
                ['solSamplesStd', 1],
                ['sampSamplesMean', 1],
                ['sampSamplesStd', 1],
                ['gpEigVals', 1],
                ['gpEigVecs', 1],
                ['gpEigValsDetailed', 1],
                ['gpEigVecsDetailed', 1],
                ['sPCA', 1],
                ['vPCA', 1],
                ['uPCA', 1],
                ['princDirDecom', 1],
                ['pcaApprox', 1],
                ['timeArray', 1],
                ['samp0Evol', 1],
                ['fc1BiasGrad', 1],
                ['fc1WeightGrad', 1],
                ['fc3BiasGrad', 1],
                ['fc3WeightGrad', 1],
                ['fullPhiGrad', 1],
                ['fc1Bias', 1],
                ['fc1Weight', 1],
                ['fc3Bias', 1],
                ['fc3Weight', 1],
                ['fullPhi', 1],
                ['numOfTestSamples', 1],
                ['psi64', 1],
                ['elbo_x', 1],
                ['elbo_y', 1],
                ['elbo_residuals', 1],
                ['normOfAbsDiff', 1],
                ['RSquared', 1],
                ['elboW', 1]
            ]
        self.plotCounter = 0
        self.displayPlots = displayPlots
        for filename in os.listdir(self.fpath):
            if filename.endswith(".png"):
                file_path = os.path.join(self.fpath, filename)
                os.remove(file_path)

   
    def useCpu(self):
        torch.set_default_dtype(torch.float64)
        torch.set_default_device('cpu')

    def producePlots(self):
        self.useCpu()
        plt.close()
        self.data = self.read()



    
        #self.plotTrueSolSurface()
        createNewCondField = self.data['createNewCondField']
        if not createNewCondField:
            self.plotxXpairs()
            #self.plotCGMap()
            self.plotRandShapeFuncs()
            self.plotResidual()
            self.plotMeanSampleAsCompetitors()

            self.plotComp3Figs()

            self.plotElbo()

            self.plotRSquaredHistory()
            
            #self.plotUncertaintySection()
        else:
            print("Plots will be created if you run the code with createNewCondField=False")
            
            #self.plotMeanRelativeErrorsSample()
            #self.plotEigenCumul()



    def save(self, writingList, fullTensors=True):
        if fullTensors == True:
            for i in range(0, len(writingList)):
                torch.save(writingList[i][1], self.path+writingList[i][0]+'.dat')

    def removeAllRefSolutions(self, folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path):
            # List all files and subdirectories in the folder
            folder_contents = os.listdir(folder_path)

            # Remove all files within the folder
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

            # Remove all subdirectories within the folder
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(folder_path)
        else:
            print(f"The folder '{folder_path}' does not exist.")

    def recreateAllRefSolFolders(self, base_path, resolution):
        # List of folder paths
        self.data = self.read()
        Nx = int(self.data['numOfTestSamples'])
        gridResolutionFenics = 'grid'+str(resolution)+'Fenics'
        gridResolutionFenicsSampleNx = 'grid'+str(resolution)+'FenicsSample'+str(Nx)
        folder_paths = [gridResolutionFenics, 'condFields', gridResolutionFenicsSampleNx]

        # Create full folder paths by joining the base path with each folder name
        folder_paths = [os.path.join(base_path, folder_name) for folder_name in folder_paths]

        # Loop over each folder path and create the folder
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def read(self):
        readingDict = {}
        for file_name in os.listdir(self.path):
            key = file_name[:-4]
            tensor = torch.load(os.path.join(self.path, file_name))
            if torch.is_tensor(tensor):
                readingDict[key] = tensor.cpu()
            elif isinstance(tensor, np.ndarray):
                readingDict[key] = torch.from_numpy(tensor).cpu()
            else:
                readingDict[key] = tensor
        return readingDict

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

 



    def plotResidual(self):
        residual = self.data['residualEvolution']

        self.plotCounter += 1
        plt.figure(self.plotCounter)
        residual = torch.FloatTensor(residual)
        minMaxCycles = torch.linspace(0, residual.size(dim=0), residual.size(dim=0))
        plt.plot(minMaxCycles, residual, '-g')
        plt.grid(True)
        plt.yscale('log')
        # plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
        plt.title("Mean Absolute Residual Convengence")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Absolute Residual")
        #plt.legend(["Residual Norm (RD)", "Moving average of the RD"])
        if not os.path.exists('./results/residuals/'):
            os.makedirs('./results/residuals/')
        plt.savefig(self.fpath+"residual.png", dpi=300, bbox_inches='tight')
        # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotElbo(self):
        elbo = self.data['elboEvolution']
        iterations = self.data['iterEvolution']
        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']
        self.plotCounter += 1
        plt.figure(self.plotCounter)
        plt.rcParams.update({'font.size': 12})
        sviCycles = torch.linspace(0, elbo.size(dim=0), elbo.size(dim=0))
        #sviCycles_elbo_x = torch.linspace(0, elbo_x.size(dim=0), elbo_x.size(dim=0))
        plt.plot(iterations, (elbo), '-m')
        #plt.plot(sviCycles, (movAvgElbo), 'c')

        plt.grid(True)
        #plt.yscale('log')
        #plt.title("ELBO")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Number of Iterations")
        plt.ylabel("ELBO")
        #plt.legend(["ELBO", "ELBO Moving Avg."])
        plt.savefig(self.fpath+"elbo.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    
    def plotVarianceConvergence(self):
        sigma = self.data['sigmaEvolution']
        varNorm = self.data['varNormEvolution']
        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']

        self.plotCounter += 1
        plt.figure(self.plotCounter)
        sviCycles = torch.linspace(0, sigma.size(dim=0), sigma.size(dim=0))
        #sviCycles_elbo_x = torch.linspace(0, elbo_x.size(dim=0), elbo_x.size(dim=0))
        plt.plot(sviCycles, (sigma), '-b')
        plt.plot(sviCycles, (varNorm), '-c')
        #plt.plot(sviCycles, (movAvgElbo), 'c')

        plt.grid(True)
        plt.yscale('log')
        plt.title("$\sigma$ and Norm($V$)")
        plt.xlabel("Number of iterations")
        plt.ylabel("$\sigma$ and Norm($V$)")
        #plt.legend(["ELBO", "ELBO Moving Avg."])
        plt.savefig(self.fpath+"sigmaConvergence.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()
    
    def plotCovarianceMatrix(self):
        cov = self.data['covMatrix'].clone().detach().cpu()
        #elbo_x = self.data['elbo_x']
        #elbo_y = self.data['elbo_y']
        #elbo_residuals = self.data['elbo_residuals']
        
        
        if cov.size(-1) > 64:
            plt.matshow(torch.log10(cov[:64, :64]), cmap='coolwarm')
        else:
            plt.matshow(torch.log10(cov[:, :]), cmap='coolwarm')
        plt.colorbar()
        plt.title("Final Covariance Matrix of the Approx. Posterior")
        plt.savefig(self.fpath+"covMatrix.png", dpi=300)
        if self.displayPlots:
            plt.show()
        else:
            plt.close()




    def plotRSquaredHistory(self):
        RSquared = self.data['RSquaredHistory']



        sviCycles = torch.linspace(0, RSquared.size(dim=0), RSquared.size(dim=0))
        RSquared = torch.where(RSquared < 0, torch.tensor(0.001), RSquared)
        plt.plot(sviCycles, RSquared, '-b')

        plt.grid(True)
        plt.title("Evolution of RSquared over many random simulations")
        plt.xlabel("Number of weighting Functions added")
        plt.ylabel("RSquared")
        plt.yscale('log')
        plt.ylim(bottom=0.1, top=1.0)
        #plt.legend(["RSquared"])
        plt.savefig(self.fpath + "RSquaredHistory.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotRelImp(self):
        relImp = self.data['relativeImprovementOfPhi']

        self.plotCounter += 1
        plt.figure(self.plotCounter)
        relImp = torch.FloatTensor(relImp)
        phiOptCycles = torch.linspace(0, relImp.size(dim=0), relImp.size(dim=0))
        plt.plot(phiOptCycles, relImp, '-m')
        plt.grid(True)
        plt.title("Relative Improvement to sqRes when GradOpt is applied for phi")
        plt.xlabel("Number of iterations")
        plt.ylabel("relImp")
        plt.legend(["relImp"])
        plt.savefig(self.fpath+"relImp.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def readRefSolution(self, x=None):
        if x is None:
            x = self.data['xPred']
        sgrid = self.data['intGrid']
        csv_data = []
        csv_dataFenics = []
        err = []
        for i in range(0, x.size(dim=0)):
            #csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
            #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
            csv_dataFenics.append(torch.load('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1)) +
                                            'Fenics/x='+"{:.1f}".format(x[i])+'.csv'))
            #err.append(torch.mean(csv_data[i]))
            #err[i] = torch.mean(abs(csv_data[i] - torch.reshape(csv_dataFenics[i], [-1])))
        return torch.stack(csv_dataFenics).detach().cpu()

    def readTestSample(self, Nx=1000):
            self.data = self.read()
            Nx = int(self.data['numOfTestSamples'])
            gridSize = self.data['intGrid'].size(1)
            rbfGridSize = self.data['rbfGrid'].size(1)
            numOfInputs = self.data['gpEigVals'].size(0)
            sol = torch.zeros(Nx, gridSize, gridSize)
            solFenics = torch.zeros(Nx, gridSize, gridSize)
            yCoeff = torch.zeros(Nx, int(rbfGridSize**2))
            cond = torch.zeros(Nx, gridSize, gridSize)
            x = torch.zeros(Nx, numOfInputs)
            #x = torch.zeros(Nx, 20) #### Needs parameterization here
            for i in range(0, Nx):
                # csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
                #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
                sol[i, :, :] = torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/sol' + "{:}".format(i) + '.csv')
                solFenics[i, :, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/solFenics' + "{:}".format(i) + '.csv'))
                yCoeff[i, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                           'FenicsSample' + str(Nx) + '/yCoeff' + "{:}".format(i) + '.csv'))
                cond[i, :, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                      'FenicsSample' + str(Nx) + '/Cond' + "{:}".format(i) + '.csv'))
                #x.append((torch.load('./model/pde/RefSolutions/grid' + str(51) +
                #                       'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv')))
                x[i, :] = (torch.load('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                  'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv'))


            return sol, solFenics, cond, x, yCoeff

    def readRefCondField(self, x=None):
        if x is None:
            x = self.data['xPred']
        sgrid = self.data['intGrid']
        csv_data = []
        csv_dataFenics = []
        err = []
        for i in range(0, x.size(dim=0)):
            #csv_data.append(torch.from_numpy(np.loadtxt('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1))+
            #                            '/x='+"{:.1f}".format(x[i])+'.csv', delimiter=',')))
            csv_dataFenics.append(torch.load('./model/pde/RefSolutions/grid'+str(sgrid.size(dim=1)) +
                                            'Fenics/Cond_x='+"{:.1f}".format(x[i])+'.csv'))
            #err.append(torch.mean(csv_data[i]))
            #err[i] = torch.mean(abs(csv_data[i] - torch.reshape(csv_dataFenics[i], [-1])))
        return torch.stack(csv_dataFenics).detach().cpu()


    def calcUncertaintyMetric(self, meanSol, meanRef, stdSol, sIndex):
        """
        :param meanSol: The mean prediction of the trained surrogate
        :param meanRef: The mean solution, calculated by a numerical solver (Reference solution)
        :param stdSol: The std at each point of the domain, calculated from actual samples from the posterior
        :param sIndex: The factor with which we multiply the standard deviation (e.g. +- 2 sigma)
        :return: The Envelope Metric
        """
        x = 1. - torch.abs(meanSol.detach().cpu() - meanRef.detach().cpu()) / (sIndex * (stdSol)+10**(-6))
        #mask = torch.logical_and(x>=-1., x<=1.)
        #torch.where(mask, torch.abs(x), 1 - torch.abs(x))
        return x

  
    def plotCGMap(self):
        dataDriven = False
        if not dataDriven:
            solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
        rbfGrid = self.data['rbfGrid']
        #yTrue = self.data['yTest']
        sgrid = self.data['intGrid']
        rbfGridSize = rbfGrid.size(dim=1)**2
        X = self.data['XCG']
        Y = self.data['YCG']
        y = self.data['yCG']
        x = self.data['xCG']
        YFenics = self.data['YCGFenics']
        yTrueProj = self.data['yProjT']
        yTrue = solFenicssamp
        
        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim))
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)))
        
        
        for i in range(0, 2):
            ymin=torch.min(yTrue[i, :, :])
            if torch.max(yTrue[i, :, :]) > torch.max(y[i, :, :]):
                ymax=torch.max(yTrue[i, :, :])
            else:
                ymax=torch.max(y[i, :, :])
            xmin=torch.min(torch.tensor(-3.))
            xmax=torch.max(torch.tensor(0.))
            combined_tensor = torch.stack((yTrue, y, yTrueProj))
            ymax = torch.max(combined_tensor[:, i, :, :])
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))


            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)))

            
            # plot x
            im1 = axs[0, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(x[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            axs[0, 0].set_title('log_10(x)')

            # plot X
            im2 = axs[0, 1].pcolormesh(xx, yy, torch.log10(X[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            axs[0, 1].set_title('log_10(X)')

            # plot Y
            im3 = axs[1, 0].pcolormesh(xx, yy, YFenics[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
            axs[1, 0].set_title('Y')

            im6 = axs[0, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            axs[0, 2].set_title('log10(xTrue)')

            im8 = axs[0, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            axs[0, 3].set_title('log10(xTrue)')

            # plot y
            im4 = axs[1, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], y[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
            axs[1, 1].set_title('y')
            # plot y
            im7 = axs[1, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], yTrueProj[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
            axs[1, 2].set_title('yTrue Projection')

            im9 = axs[1, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], yTrue[i, :, :], cmap='coolwarm', vmax=ymax, vmin=ymin)
            axs[1, 3].set_title('yTrue')

            # add colorbars
            cbar1 = fig.colorbar(im1, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]])
            cbar2 = fig.colorbar(im4, ax=[axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]])
            
            cbar1.ax.set_position([0.8, 0.55, 0.03, 0.3])
            cbar2.ax.set_position([0.8, 0.15, 0.03, 0.3])

            plt.savefig(self.fpath + "MappingPlots"+str(i)+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        tess = 't'

    def plotRandShapeFuncs(self):
        dataDriven = False
        if not dataDriven:
            solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
        rbfGrid = self.data['rbfGrid']
        #yTrue = self.data['yTest']
        sgrid = self.data['intGrid']
        rbfGridSize = rbfGrid.size(dim=1)**2
        X = self.data['XCG']
        Y = self.data['YCG']
        y = self.data['yCG']
        x = self.data['xCG']
        YFenics = self.data['YCGFenics']
        yTrueProj = self.data['yProjT']
        yTrue = solFenicssamp
        
        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim))
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)))
        randShapeFuncs = self.data['randShapeFuncs']
        
        for i in range(0, 1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)))
            xx, yy = torch.meshgrid(torch.linspace(0, 1, randShapeFuncs.size(-2)), torch.linspace(0, 1, randShapeFuncs.size(-1)))
            axs.set_xticks([])
            axs.set_yticks([])
            # plot x
            im1 = axs.pcolormesh(sgrid[0, :, :], sgrid[1, :, :], randShapeFuncs, cmap='inferno')

            plt.savefig(self.fpath + "randShapeFuncs"+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        tess = 't'



    def plotxXpairs(self):
        dataDriven = False
        if not dataDriven:
            solsamp, solFenicssamp, condsamp, xsamp, yCoeffsamp = self.readTestSample()
            solsamp = solsamp.detach().cpu()
            solFenicssamp = solFenicssamp.detach().cpu()
            condsamp = condsamp.detach().cpu()
            xsamp = xsamp.detach().cpu()
            yCoeffsamp = yCoeffsamp.detach().cpu()

        rbfGrid = self.data['rbfGrid']
        #yTrue = self.data['yTest']
        sgrid = self.data['intGrid']
        rbfGridSize = rbfGrid.size(dim=1)**2
        X = self.data['XCG']
        Y = self.data['YCG']
        y = self.data['yCG']
        x = self.data['xCG']
        YFenics = self.data['YCGFenics']
        yTrueProj = self.data['yProjT']
        yTrue = solFenicssamp.clone().detach().cpu()
        
        self.reducedDim = 9
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, sgrid.size(-1)), torch.linspace(0, 1, sgrid.size(-1)), indexing='ij')
        
        
        for i in range(0, 1):
            ymin=torch.min(yTrue[i, :, :])
            if torch.max(yTrue[i, :, :]) > torch.max(y[i, :, :]):
                ymax=torch.max(yTrue[i, :, :])
            else:
                ymax=torch.max(y[i, :, :])
            xmin=torch.min(torch.tensor(-3.))
            xmax=torch.max(torch.tensor(0.))
            combined_tensor = torch.stack((yTrue, y, yTrueProj))
            ymax = torch.max(combined_tensor[:, i, :, :])
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))


            xx, yy = torch.meshgrid(torch.linspace(0, 1, X.size(-2)), torch.linspace(0, 1, X.size(-1)), indexing='ij')

            
            # plot x
            im1 = axs[0, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 0].set_title('log_10(x)')

            # plot X
            im2 = axs[0, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+1, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 1].set_title('log_10(X)')

            im6 = axs[0, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+2, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 2].set_title('log10(xTrue)')

            im8 = axs[0, 3].pcolormesh(sgrid[0, :, :], sgrid[1, :, :], torch.log10(condsamp[i+3, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[0, 3].set_title('log10(xTrue)')

            # plot Y
            im3 = axs[1, 0].pcolormesh(xx, yy, torch.log10(X[i, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 0].set_title('Y')

            # plot y
            im4 = axs[1, 1].pcolormesh(xx, yy, torch.log10(X[i+1, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 1].set_title('y')
            # plot y
            im7 = axs[1, 2].pcolormesh(xx, yy, torch.log10(X[i+2, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 2].set_title('yTrue Projection')

            im9 = axs[1, 3].pcolormesh(xx, yy, torch.log10(X[i+3, :, :]), cmap='viridis', vmax=xmax, vmin=xmin)
            #axs[1, 3].set_title('yTrue')

            # add colorbars
            cbar1 = fig.colorbar(im1, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]])
            #cbar2 = fig.colorbar(im4, ax=[])
            
            cbar1.ax.set_position([0.78, 0.15, 0.03, 0.6])
            cbar1.set_label('$log_{10}(x)$',  fontsize=16)
            #cbar2.ax.set_position([0.8, 0.15, 0.03, 0.3])

            plt.savefig(self.fpath + "xXpairs"+str(i)+".png", dpi=300, bbox_inches='tight')
            if self.displayPlots:
                plt.show()
            else:
                plt.close()
        tess = 't'



    def gpExpansionExponentialParallel(self, x, gpEigVals, gpEigVecs, sgrid, fraction=0.):

        out = torch.einsum('i,...i,ij->...j', torch.sqrt(gpEigVals),
                                          x, gpEigVecs)
        out = torch.reshape(out, (*x.size()[:-1], sgrid.size(dim=1), sgrid.size(dim=1)))
        mask = out > fraction
        out = torch.where(mask, torch.tensor(0.1), torch.tensor(1.))
        #outNumpy = torch.reshape(out, [100, 32, 32]).cpu().numpy()
        return out


    def plotUncertaintySection(self, sIndex=2):
        sgrid = self.data['intGrid']
        meanAbsError = self.data['meanAbsError']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        yFENICSTrue = self.data['yFENICSTrue']
        RSquared = self.data['RSquaredHistory'][-1]
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            RSquared = calcRSquared(yTrue, yPred)

        yTrue = self.data['yProjT']
        #
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)

        columns = 5
        fig, axs = plt.subplots(columns, 2, figsize=(3 * columns, (3)*columns), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):
            vmin = torch.min(yTrue[i, :, :])
            vmax = torch.max(yTrue[i, :, :])
            axs[0, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 yPred[i, :, :], cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
            axs[1, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 yTrue[i, :, :], cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
            axs[2, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 EnvMetric[i, :, :], cmap='coolwarm', shading='auto', vmin=-1., vmax=1.)
            axs[3, i].plot(torch.linspace(0, 1.4142, yPred.size(-1)), torch.diag(yTrue[i, :, :]), 'r')
            axs[3, i].plot(torch.linspace(0, 1.4142, yPred.size(-1)), torch.diag(yPred[i, :, :]), '--b')
            axs[3, i].fill_between(torch.linspace(0, 1.4142, yPred.size(-1)), torch.diag(yPred[i, :, :]) + torch.diag(sIndex*sampSamplesStd[i, :, :]),
                                                                                          torch.diag(yPred[i, :, :]) - torch.diag(sIndex*sampSamplesStd[i, :, :]),
                                                                                          facecolor='blue', alpha=0.3)
            axs[3, i].grid(True)
            axs[4, i].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[i, :, :], cmap='jet', shading='auto')
            axs[0, i].set_title("Predicted Solution for Sample " + "{:.1f}".format(i))
            axs[0, i].set_aspect('equal')
            axs[1, i].set_title("True Solution for Sample" + "{:.1f}".format(i))
            axs[1, i].set_aspect('equal')
            axs[2, i].set_title("Envelope Metric" + "{:.1f}".format(i))
            axs[2, i].set_aspect('equal')
            axs[3, i].set_title("Uncertainty Section " + "{:.1f}".format(i))
            #axs[3, i].set_aspect('equal')
            axs[4, i].set_title("Conductivity for Sample " + "{:.1f}".format(i))
            axs[4, i].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[1, i].collections[0], orientation='horizontal')
            fig.colorbar(axs[2, i].collections[0], orientation='horizontal')
            fig.colorbar(axs[4, i].collections[0], orientation='horizontal')
            relPANIS = calcEpsilon(yTrue=yTrue, yPred=yPred)

            yPINO = yTrue + torch.randn(yTrue.size())
            PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPINO_pureFNOVF50ttt.pt"
            if os.path.exists(PINOsavePath):
                yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)
            else:
                print("PINOsavePath doesn't exist! Loading a random tensor instead!")
            
            relPANIS = calcEpsilon(yTrue=yFENICSTrue, yPred=yPred)
            relPINO = calcEpsilon(yTrue=yFENICSTrue, yPred=yPINO)

            fig.suptitle('For 100 different Samples $R^2$ = ' + f'{RSquared.item():.4f}' +
             ' and EnvelopScore = ' + f'{EnvelopScore.item():.4f}' + 
             ' (Target = 0.95)' + ' meanAbsError: ' + f'{meanAbsError.item():.4f}' + 
             '\n relPANIS: ' + f'{relPANIS.item():.4f}' + 
             '\n relPINO: ' + f'{relPINO.item():.4f}')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "UncertaintySection.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotMeanSampleAsCompetitors(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        yFENICSTrue = self.data['yFENICSTrue'].detach().clone().cpu()
        RSquared = self.data['RSquared'][-1].clone().detach()
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
        RSquared = calcRSquared(yFENICSTrue, yPred)
            
        yPINO = yTrue + torch.randn(yTrue.size())
        PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPINOmVF50.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yFENICSTrue, yPINO)
        relativeL2ErrorPANIS = calcEpsilon(yTrue=yTrue, yPred=yPred)
        relativeL2ErrorPINO = calcEpsilon(yTrue=yTrue, yPred=yPINO)
        print('PINO relative L2 error: '+f'{relativeL2ErrorPINO.item():.5f}')
        print('PANIS relative L2 error: '+f'{relativeL2ErrorPANIS.item():.5f}')
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])

        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            j = 30 + i
          
            

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yFENICSTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yFENICSTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanTrueFENICS = yFENICSTrue[j][yFENICSTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            #axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueFENICS, 'c')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 1].set_title("Error FNO" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')
            axs[0, 2].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Ground-truth", "FNO", "PANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 3].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 5].collections[0], orientation='vertical')
            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithCompetitors.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()

    def plotMeanSampleAsMyself(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        RSquared = torch.tensor(self.data['RSquared'][-1])
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        #PINOsavePath = "./PANISonl005mean.dat"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            j =  51 + i 

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            #axs = fig.add_subplot(3, 4, i * 4 + 5)
            #axs[i, 4].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
            #                     EnvMetric[i, :, :], cmap='coolwarm', shading='auto', vmin=-1., vmax=1.)
            #axs = fig.add_subplot(3, 4, i * 4 + 6)
            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 2].set_title("Error with $\mathbf{y}_F$" + ' $R^2$=' +f'{RSquared.item():.3f}')
            axs[0, 1].set_title("Error without $\mathbf{y}_F$" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Ground-truth", "Our Model without $\mathbf{y}_F$", "Our Model with $\mathbf{y}_F$"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 3].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 5].collections[0], orientation='vertical')
            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithMyself.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotMeanSampleAsPANIS(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd']
        yFENICSTrue = self.data['yFENICSTrue']
        RSquared = torch.tensor(self.data['RSquared'][-1])
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        PINOsavePath = "./PANISonl005mean.dat"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
            yPINOstd = torch.load("./PANISonl005sigma.dat").detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        #yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 4, figsize=(20, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):


            j = 30 + i 



            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.
            vmax = torch.max(torch.abs(yPINO[j] - yTrue[j]))


            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 2].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            
            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPINO[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagMean2 = yPINO[j][yPINO[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagStd2 = yPINOstd[j][yPINO[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanTrueFenics = yFENICSTrue[j][yFENICSTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrueFenics, 'c')
            axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 3].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 3].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 3].fill_between(torch.linspace(0, 1., yPINO.size(-1)), diagMean2 + diagStd2,
                                                                                          diagMean2 - diagStd2,
                                                                                          facecolor='green', alpha=0.3)
            axs[i, 3].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")

            axs[0, 2].set_title("Error mPANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')
            axs[0, 1].set_title("Error PANIS" + ' $R^2$=' +f'{RSquaredPINO.item():.3f}')

            axs[0, 3].set_title('Solution Slice')

            axs[i, 3].legend(["Full Ground-truth", "CG Ground-truth", "PANIS", "mPANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 2].collections[0], orientation='vertical')


        plt.tight_layout()
        plt.savefig(self.fpath + "comparingWithPANIS.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    def plotComp3Figs(self, sIndex=2):
        sgrid = self.data['intGrid']
        sampSamplesMean = self.data['sampSamplesMean']
        sampSamplesStd = self.data['sampSamplesStd'].clone().detach()
        RSquared = self.data['RSquared'][-1].clone().detach()
        intPoints = sgrid.size(dim=1)
        yPred = sampSamplesMean.detach().clone().cpu()
        
        outOfDistributionPrediction = self.data['outOfDistributionPrediction']
        dataDriven = outOfDistributionPrediction
        if not dataDriven:
            sol, solFenics, cond, x, yCoeff = self.readTestSample()
            yTrue = torch.reshape(solFenics, [solFenics.size(dim=0), intPoints, -1]).detach().clone().cpu()
        else:
            sol = self.data['yTest']
            cond = self.data['xTest']
            yTrue = sol[:yPred.size(0)].detach().clone().cpu()
            #RSquared = calcRSquared(yTrue[:, 3:-3, 3:-3], yPred[:, 3:-3, 3:-3])
        RSquared = calcRSquared(yTrue, yPred)
            
        yTrue = self.data['yProjT']
        yPINO = yTrue + torch.randn(yTrue.size())
        #PINOsavePath = "/home/matthaios/Projects/pino/checkpoints/yPI.pt"
        PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/CGnoyFdddddddddd.pt"
        #PINOsavePath = "/home/matthaios/PredictionData/lengthScale005_without_yF/yPI.pt"
        if os.path.exists(PINOsavePath):
            yPINO = torch.load(PINOsavePath).detach().clone().to(yTrue.device)[:yTrue.size(0)]
        else:
            print("PINOsavePath doesn't exist! Loading a random tensor instead!")
        RSquaredPINO = calcRSquared(yTrue, yPINO)
        yPINOstd = torch.std(torch.abs(yPINO-yTrue), dim=[-1, -2])
        yPINOmean = torch.mean(torch.abs(yPINO-yTrue), dim=[-1, -2])
        

        if True:
            EnvMetric = []
            for i in range(0, yPred.size(dim=0)):
                EnvMetric.append(self.calcUncertaintyMetric(meanSol=yPred[i, :, :], meanRef=yTrue[i, :, :],
                                                            stdSol=sampSamplesStd[i, :, :], sIndex=sIndex))
            EnvMetric = torch.stack(EnvMetric)
            EnvelopScore = torch.where(EnvMetric >= 0., torch.tensor(1.), torch.tensor(0.))
            EnvelopScore = torch.mean(EnvelopScore, dim=0)
            EnvelopScore = torch.mean(EnvelopScore, dim=(0, 1))
        else:
            EnvelopScore = torch.tensor(1.)


        plt.rcParams.update({'font.size': 16})

        columns = 2
        fig, axs = plt.subplots(columns, 3, figsize=(16, 8), num=self.plotCounter)
        c_x = cond
        cax = []
        for i in range(2):

            j =  73 + 8*i

            if torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j])) > torch.max(torch.abs(yPred[j] - yTrue[j])):
                vmax = torch.max(torch.abs(yPINOmean[j] + 2* yPINOstd[j]))
            else:
                vmax = torch.max(torch.abs(yPred[j] - yTrue[j]))

            if torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j])) < torch.min(torch.abs(yPred[j] - yTrue[j])):
                vmin = torch.min(torch.abs(yPINOmean[j] - 2* yPINOstd[j] - yTrue[j]))
            else:
                vmin = torch.min(torch.abs(yPred[j] - yTrue[j]))
            vmin=0.



            #axs = fig.add_subplot(3, 4, i * 4 + 1)
            axs[i, 0].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 c_x[j, :, :], cmap='jet', shading='auto')

            axs[i, 1].pcolormesh(sgrid[0, :, :], sgrid[1, :, :],
                                 torch.abs(yPred[j, :, :]-yTrue[j, :, :]), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
            

            diag2CG = yPred[j].flip(0).diagonal()
            diag1CG = torch.diag(yPred[j, :, :])
            diag1CGStd =  torch.diag(sIndex*sampSamplesStd[j, :, :])
            diag2CGStd =  sampSamplesStd[j].flip(0).diagonal() * sIndex
            diagMean = yPred[j][yPred[j].size(0)//2, :]
            diagStd = sampSamplesStd[j][yPred[j].size(0)//2, :] * sIndex
            diagMeanTrue = yTrue[j][yTrue[j].size(0)//2, :] #yTrue[j].flip(0).diagonal()
            diagMeanPINO = yPINO[j][yPINO[j].size(0)//2, :] #yPINO[j].flip(0).diagonal()
            axs[i, 2].plot(torch.linspace(0, 1., yPred.size(-1)), diagMeanTrue, 'r')
            #axs[i, 3].plot(torch.linspace(0, 1., yPINO.size(-1)), diagMeanPINO, '--g')
            axs[i, 2].plot(torch.linspace(0, 1., yPred.size(-1)), diagMean, '--b')
            
            axs[i, 2].fill_between(torch.linspace(0, 1., yPred.size(-1)), diagMean + diagStd,
                                                                                          diagMean - diagStd,
                                                                                          facecolor='blue', alpha=0.3)
            axs[i, 2].grid(True)
            
            axs[0, 0].set_title("Input $\mathbf{x}$")
            axs[0, 1].set_title("Error PANIS" + ' $R^2$=' +f'{RSquared.item():.3f}')

            axs[0, 2].set_title('Solution Slice')

            axs[i, 2].legend(["Ground-truth", "PANIS"])
            #axs[i, 5].set_aspect('equal')
            #cax.append(fig.add_axes([(1 - 0.1) / (6-1) * (i + 1), 0.15, 0.02, 0.80]))
            fig.colorbar(axs[i, 0].collections[0], orientation='vertical')
            #fig.colorbar(axs[i, 1].collections[0], orientation='vertical')
            fig.colorbar(axs[i, 1].collections[0], orientation='vertical')

            
            #fig.suptitle('Comparison on validation dataset with PINO')
        # cax, kw = plt.colorbar.make_axes([ax for ax in axs.flat])

        #fig.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(self.fpath + "comp3figs.png", dpi=300, bbox_inches='tight')
        if self.displayPlots:
            plt.show()
        else:
            plt.close()


    