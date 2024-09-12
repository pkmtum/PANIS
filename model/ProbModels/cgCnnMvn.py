import numpy as np
import torch
from utils.nnArchitectures.nnCoarseGrained import nnPANIS, nnmPANIS




class probabModel:
    def __init__(self, pde, stdInit=2, display_plots=False, lr=0.001, sigma_r=0, yFMode=True, randResBatchSize=None, reducedDim=None, dataset=None):  # It was stdInit=8
        self.pde = pde
        self.sigma_r = sigma_r
        self.yFMode = yFMode
        self.nele = pde.nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_counter = 0
        self.guideExecCounter = 0
        self.modelExecCounter = 0
        self.dataDriven = True
        self.dataDrivenFlag = False

        self.lowUnif = -1.
        self.highUnif = 1.
        self.readData = 0
        self.xPoolSize = 1
        if self.readData == 1:
            print("Pool of input data x was read from file.")
        else:
            print("Pool of input data x generated for this run.")
            self.data_x = torch.normal(self.mean_px, self.sigma_px, size=(self.xPoolSize,))
            self.data_x = 4 * self.sigma_px * torch.rand(size=(self.xPoolSize,)) - 2 * self.sigma_px
            print(self.data_x)
            print(torch.exp(-self.data_x))
            self.data_x = torch.linspace(-1., 1., self.xPoolSize)


        self.constant_phi_max = torch.ones((pde.NofShFuncs, pde.NofShFuncs))
        self.phi_max = torch.rand((pde.NofShFuncs, 1), requires_grad=True) * 0.01 + torch.ones(pde.NofShFuncs, 1)
        self.phi_max = self.phi_max / torch.linalg.norm(self.phi_max)

        phi_max_leaf = self.phi_max.clone().detach().requires_grad_(True)
        self.phi_max = phi_max_leaf
        self.phiBase = torch.reshape(self.phi_max, [1, -1])

        self.phi_max_history = np.zeros((pde.NofShFuncs, 1))
        self.sigma_history = np.zeros((pde.NofShFuncs, 1))
        self.temp_res = []
        self.full_temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0
        self.stdInit = stdInit
        self.randResBatchSize = randResBatchSize
        self.compToKeepInPCA = 4
        self.residualCorrector = torch.sqrt(torch.tensor(self.Nx_samp))

        self.validationIndex = 0


        self.reducedDim = reducedDim
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, self.pde.intPoints), torch.linspace(0, 1, self.pde.intPoints), indexing='ij')
        self.xtoXCnn = None
        self.neuralNetCG = None
        self.NTraining = 99
        self.I = torch.ones(self.pde.NofShFuncs)*(stdInit)
        self.I.requires_grad_(True)
        self.vDim = 10
        if yFMode:
            self.V = torch.ones((self.reducedDim**2, self.vDim))*(-3) +torch.randn(self.reducedDim**2, self.vDim)/10
        else:
            self.V = torch.ones((self.pde.NofShFuncs, self.vDim))*(-3) +torch.randn(self.pde.NofShFuncs, self.vDim)/10
        self.V.requires_grad_(True)

        self.globalSigma = torch.tensor([float(stdInit)])
        self.globalSigma.requires_grad_(True)
        if yFMode:
            self.neuralNet = nnmPANIS(reducedDim=self.reducedDim, cgCnn=self.neuralNetCG, xtoXCnn=self.xtoXCnn, pde=pde, extraParams=[self.V, self.globalSigma, yFMode])
        else:
            self.neuralNet = nnPANIS(reducedDim=self.reducedDim, cgCnn=self.neuralNetCG, xtoXCnn=self.xtoXCnn, pde=pde, extraParams=[self.V, self.globalSigma, yFMode])

        
        numOfPars = self.neuralNet.count_parameters()
        print("Number of NN Parameters Used: ", numOfPars)

        
        

        self.losss = torch.nn.MSELoss(reduction='mean')
        

        if True:
            self.xConst = torch.reshape(torch.randn(self.NTraining+1, self.pde.numOfInputs),
                                       [self.NTraining+1, 1, self.pde.numOfInputs])

        else:
            self.xConst = pde.sampX.view(pde.sampX.size(0), 1, pde.sampX.size(-1))

        if self.yFMode:
            self.yF = (torch.ones(self.pde.shapeFunc.aInvBCO.size(-1)) + 0.1 * torch.rand(self.pde.shapeFunc.aInvBCO.size(-1))).repeat(self.NTraining+1, 1).requires_grad_(True)
            
        if not self.yFMode:
            self.manualOptimizer = torch.optim.Adam(params=list(self.neuralNet.parameters())+ [self.V] + [self.globalSigma], lr=lr, maximize=True, amsgrad=False)
            
        else:
            
            self.manualOptimizer = torch.optim.Adam(params=list(self.neuralNet.parameters())+ [self.neuralNet.V] + [self.neuralNet.globalSigma] + [self.yF], lr=lr, maximize=True, amsgrad=False)
           
        self.ELBOLossHistory = torch.zeros(1, 1000)
       
        




    def sampleFromQxyMVN(self, Nx, useTheRealNN=True):
        
        if not self.dataDrivenFlag:
            x = torch.reshape(torch.randn(Nx, self.pde.numOfInputs),
                                       [Nx, 1, self.pde.numOfInputs])

            if self.yFMode:
                indices = torch.randperm(self.xConst.size(0))[:Nx]
                x = self.xConst[indices]
                y = self.neuralNet.forward(x) + torch.einsum('...ij,...j->...i', self.pde.shapeFunc.aInvBCO, 1 * self.yF[indices]).view(x.size(0), 1, -1)
            else:
                y = self.neuralNet.forward(x) + torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(Nx, 1, self.vDim)) + torch.pow(10, self.globalSigma/2.) * torch.randn(Nx, 1, self.pde.NofShFuncs)
            
            y = y 
        return x, y
    

    
 
    
    
    def entropyMvnShortAndStable(self, V, sigma):
        manualEntropyStable = 0.5 * torch.logdet(torch.einsum('ij,kj->ik', V, V)+ sigma**2 * torch.eye(self.pde.NofShFuncs))
        return manualEntropyStable
    
    def entropyMvnShortAndStableNN(self):
        manualEntropyStable = 0.5 * torch.logdet(torch.einsum('ij,kj->ik', torch.pow(10, self.neuralNet.V), torch.pow(10, self.neuralNet.V)) \
                                                 + torch.pow(10, self.neuralNet.globalSigma/2)**2 * torch.eye(self.reducedDim**2))

        return manualEntropyStable
    
    
    
    
    def logProbMvnUninformative(self, x):
        manualLogProbShortest = - 0.5 * torch.einsum('...i,...i->...', x, x) * 10**(-8)
        return manualLogProbShortest
    

    
    def sviStep(self):
        MultipleActiveClasses = False
        self.manualOptimizer.zero_grad()
        
        x, y = self.sampleFromQxyMVN(self.Nx_samp)
        indices1 = torch.randperm((self.pde.shapeFuncsDim+1)**2)[:self.randResBatchSize]
        phi = torch.eye(((self.pde.shapeFuncsDim+1)**2)).unsqueeze(0).expand(x.size(0), -1, -1)[:, indices1, :]
        if MultipleActiveClasses:
            indices2 = torch.randperm(49)[:10]
            phi2 = torch.eye(49).unsqueeze(0).expand(x.size(0), -1, -1)[:, indices2, :]
        self.phi = phi
        

        if self.yFMode:
            entropy = self.entropyMvnShortAndStableNN()
        else:
            entropy = self.entropyMvnShortAndStable(torch.pow(10, self.V), torch.pow(10, self.globalSigma/2))
        logProb = torch.mean(self.logProbMvnUninformative(y))
        res = self.pde.calcSingleResGeneralParallel(x, y, phi)
        if MultipleActiveClasses:
            res2 = self.pde.calcSingleResGeneralParallel2(x, y, phi2)
            res = torch.cat((res, res2), dim=1)

        absRes = torch.abs(res)
        likelihood = - 1./2./self.sigma_r * torch.sum(torch.mean(absRes, dim=0), dim=0)
        yField = self.pde.shapeFunc.cTrialSolutionParallel(y)


        loss = likelihood + logProb + entropy
        loss.backward(retain_graph=True)
        self.manualOptimizer.step()
        if res.size(0) > 1 or self.Nx_samp == 1:
            self.temp_res.append(torch.mean(torch.abs(res)))

        self.neuralNet.iter = self.neuralNet.iter + 1

        return loss.clone().detach()



    def removeSamples(self):
        self.temp_res = []
        self.full_temp_res = []




    
    def samplePosteriorMvn(self, x, Nx=1, Navg=1): 
        
        xReshaped = torch.reshape(x, [x.size(0), 1, -1])

        dimOut = self.pde.NofShFuncs

        if Nx <= 10:
            if self.yFMode:
                mean = self.neuralNet.forwardMultiple(xReshaped, Navg=Navg)
            else:
                mean = self.neuralNet.forward(xReshaped).view(Nx, 1, -1).repeat(1, Navg, 1)
    
            
        else:
            NN = int(Nx/10)
            mean = torch.zeros(Nx, Navg, dimOut)
            for i in range(0, NN):
                if self.yFMode:
                    mean[10*i:10*i+10] = self.neuralNet.forwardMultiple(xReshaped[10*i:10*i+10], Navg=Navg)
                else:
                    mean[10*i:10*i+10] = self.neuralNet.forward(xReshaped[10*i:10*i+10]).view(10, 1, -1).repeat(1, Navg, 1)
                
        if self.yFMode:
            ySamples = mean
        else:
            Sigma = torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(mean.size(0), Navg, self.vDim)) + \
              torch.pow(10, self.globalSigma/2.) * torch.randn(mean.size(0), Navg, mean.size(-1))
            ySamples = mean + Sigma

        
        return ySamples, mean
    
    def calcCovarianceMatrix_III(self):
        V = torch.pow(10, self.V)
        globalSigma = torch.pow(10, self.globalSigma)
        cov = V @ V.t() + torch.diag(globalSigma)
        return cov

    def samplePosteriorMean(self, x):
        if self.pde.numOfInputs == 1:
            x = torch.reshape(x, [1, 1, -1])
        elif self.pde.numOfInputs > 1:
            x = torch.ones(self.pde.numOfInputs) * x
            x = torch.reshape(x, [1, 1, -1])
        res = self.neuralNet.forward(x)
        if self.doPCA:
            res = self.pde.meanSampYCoeff + torch.einsum('ji,i->j', self.usedEigVecs,
                                                         res)
        return res






