### Importing Libraries ###
import torch
import os
from utils.powerIteration import powerIterTorch
import model.pde.shapeFunctions.hatFunctions as hatFuncs
from model.pde.numIntegrators.trapezoid import trapzInt2D, trapzInt2DParallel, simpsonInt2DParallel
import matplotlib.pyplot as plt
from model.pde.pdeTrueSolFenics import solve_pde, create_map
from input import device
from utils.variousFunctions import calcRSquared, createFolderIfNotExists
from torch.utils.data import TensorDataset



class pdeForm:
    def __init__(self, nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp, createNewCondField, device, post, rhs=None, reducedDim=None, options=None):
        self.shapeFuncsDim = shapeFuncsDim
        self.nele = nele
        self.grid_x, self.grid_y = torch.meshgrid(torch.linspace(0, 1, nele + 1), torch.linspace(0, 1, nele + 1),
                                                  indexing='ij')
        self.grid = torch.stack((self.grid_x, self.grid_y), dim=0)
        self.gridW_x, self.gridW_y = torch.meshgrid(torch.linspace(0, 1, shapeFuncsDim + 1), torch.linspace(0, 1, shapeFuncsDim + 1),
                                                  indexing='ij')
        self.gridW = torch.stack((self.gridW_x, self.gridW_y), dim=0)
        self.gridW_x2, self.gridW_y2 = torch.meshgrid(torch.linspace(0, 1, 6 + 1), torch.linspace(0, 1, 6 + 1),
                                                  indexing='ij')
        self.gridW2 = torch.stack((self.gridW_x2, self.gridW_y2), dim=0)
        self.NofShFuncs = torch.reshape(self.grid_x, [-1]).size(dim=0)
        self.NofShFuncsW = torch.reshape(self.gridW_x, [-1]).size(dim=0)
        self.node_corrs = torch.reshape(self.grid, [2, -1])
        self.dl = 1 / nele
        self.dlW = 1 / shapeFuncsDim
        self.options = options
        self.optionsMod = {'alpha': options['alpha'], 'u0': options['u0']}
        self.intPoints = options['integrationGrid'] # it needs to be always 8 multiple + 1
        self.createNewCondField = createNewCondField
        if self.createNewCondField:
            post.removeAllRefSolutions(folder_path='./model/pde/RefSolutions/')
            post.recreateAllRefSolFolders(base_path='./model/pde/RefSolutions/', resolution=self.intPoints)
        self.sgrid_x, self.sgrid_y = torch.meshgrid(torch.linspace(0, 1, self.intPoints),
                                                    torch.linspace(0, 1, self.intPoints),
                                                    indexing='ij')
        self.sgrid = torch.stack((self.sgrid_x, self.sgrid_y), dim=0)
        if self.intPoints > 100:
            self.gpgrid_x, self.gpgrid_y = torch.meshgrid(torch.linspace(0, 1, 100),
                                                    torch.linspace(0, 1, 100),
                                                    indexing='ij')
        else:
            self.gpgrid_x, self.gpgrid_y = torch.meshgrid(torch.linspace(0, 1, self.intPoints),
                                                    torch.linspace(0, 1, self.intPoints),
                                                    indexing='ij')
        self.gpgrid = torch.stack((self.gpgrid_x, self.gpgrid_y), dim=0)
        self.savePath = './model/pde/RefSolutions/condFields/'
        #self.shapeFunc = hatFuncs.ChebyshevInterpolation(self.grid, self.sgrid, rhs, uBc=options['boundaryCondition'], reducedDim=reducedDim)
        self.shapeFunc = hatFuncs.rbfInterpolation(self.grid, self.gridW, self.gridW2, self.sgrid, 1 / self.dl ** 2, 1 / self.dlW ** 2, rhs, uBc=options['boundaryCondition'], reducedDim=reducedDim, savePath=self.savePath, options=self.optionsMod)

        self.mean_px = mean_px
        self.sigma_px = sigma_px
        self.sigma_r = sigma_r
        self.Nx_samp = Nx_samp
        self.rhs = rhs
        self.effective_nele = None
        self.f = None
        self.s = torch.linspace(0, 1, nele + 1)
        self.systemRhs = None
        self.numOfInputs = options['inputDimensions']  # It was 20 before the PCA plots #32
        if options['boundaryCondition'] == 'sinx':
            self.uBc = torch.sin(torch.linspace(0, 2*torch.pi, self.shapeFunc.bcNodes.size(0)))*5. + 10.
            self.uBc[-1] = self.uBc[0]
        else:
            self.uBc= options['boundaryCondition']
        self.alphaPDE = 0.
        self.u0PDE = options['u0']
        if self.options['alpha'] != 0:
            self.Linear = False
        else:
            self.Linear = True
        # l=100. and s=0.14 gives similar results like exp(-1), exp(1)
        self.lengthScale = options['lengthScale']  # 0.2 before 07.09.23, 0.1 before 01.02.23 (l=0.25 with squared distance is acceptable)
        self.gpSigma = 1.  # 0.35 before 07.09.23
        
        
        if options['volumeFraction']=='FR50':
            self.fraction = 0
        elif options['volumeFraction']=='FR74':
            self.fraction = 1.3
        elif options['volumeFraction']=='FR00':
            self.fraction = -5.
        elif options['volumeFraction']=='FR10':
            self.fraction = -1.28
        elif options['volumeFraction']=='FR20':
            self.fraction = -0.84
        elif options['volumeFraction']=='FR30':
            self.fraction = -0.52
        elif options['volumeFraction']=='FR40':
            self.fraction = -0.25
        elif options['volumeFraction']=='FR60':
            self.fraction = 0.26
        elif options['volumeFraction']=='FR70':
            self.fraction = 0.53
        elif options['volumeFraction']=='FR80':
            self.fraction = 0.85
        elif options['volumeFraction']=='FR90':
            self.fraction = 1.29
        elif options['volumeFraction']=='FR100':
            self.fraction = 5.00
        if options['contrastRatio']=='CR10':
            self.phaseHigh = torch.tensor(1./10.)
        elif options['contrastRatio']=='CR3':
            self.phaseHigh = torch.tensor(1./3.)
        elif options['contrastRatio']=='CR50':
            self.phaseHigh = torch.tensor(1./50.)
        self.phaseLow = torch.tensor(1.)
        if options['modeType']=='test':
            torch.manual_seed(1)
        elif options['modeType']=='train':
            torch.manual_seed(0)

        self.idx = create_map(options['refSolverIntGrid'])
         

        
        if self.createNewCondField:
            self.gpCov, self.gpEigVals, self.gpEigVecs = self.doGpForInputs(self.gpgrid, self.lengthScale, self.gpSigma,
                                                                            self.numOfInputs)
            self.gpCovDetailed, self.gpEigValsDetailed, self.gpEigVecsDetailed = self.doGpForInputs(self.gpgrid,
                                                                                                    self.lengthScale,
                                                                                                    self.gpSigma, self.numOfInputs+20)
            torch.save(self.gpCov, self.savePath + 'gpCov.dat')
            torch.save(self.gpEigVals, self.savePath + 'gpEigVals.dat')
            torch.save(self.gpEigVecs, self.savePath + 'gpEigVecs.dat')
            torch.save(self.gpCovDetailed, self.savePath + 'gpCovDetailed.dat')
            torch.save(self.gpEigValsDetailed, self.savePath + 'gpEigValsDetailed.dat')
            torch.save(self.gpEigVecsDetailed, self.savePath + 'gpEigVecsDetailed.dat')
        else:
            self.gpEigVals = torch.load(self.savePath + 'gpEigVals.dat').to(device)
            self.gpEigVecs = torch.load(self.savePath + 'gpEigVecs.dat').to(device)
            self.sampX = torch.load(self.savePath + 'sampX.dat').to(device)
        



    
    ### Do Gaussian Process and get the Covariance Matrix, Eigenvalues and Eigenvectors ###
    def doGpForInputs(self, x, l=1.0, sigma=1.0, numOfEigsToKeep=10, calcFullEig=False):
        """
        Attention!: Currently the full covariance Matrix/Eigenvalues/Eigenvectors are stored. This isn't optimal and it
        could cause problem in the future. We should probably discard the biggest part of these matrices.
        l: Correlation length or Length-scale of the RBF kernel for the GP.
        sigma: Amplitude of the RBF kernel for the GP.
        :return: The Covariance Matrix of the GP, the Eigenvalues and the Eigenvectors of the Covariance Matrix.
        """

        # Euclidean distance between x and y
        dist = torch.cdist(torch.reshape(x, [2, -1]).T, torch.reshape(x, [2, -1]).T, p=2)
        cov = sigma ** 2 * torch.exp(-dist**2/l**2)
        #covNumpy = cov.cpu().numpy()
        if calcFullEig == True:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.flip(eigenvalues, dims=[0])
            eigenvectors = torch.flip(eigenvectors, dims=[1])
        else:
            eigenvalues, eigenvectors = powerIterTorch(cov, num_eigenvalues=numOfEigsToKeep, max_iterations=100)
            eigenvalues = torch.tensor(eigenvalues).to(device)
            eigenvalues = torch.where(eigenvalues < torch.tensor(0.), 0, eigenvalues)
            eigenvectors = torch.stack(eigenvectors)


        return cov, eigenvalues, eigenvectors

    ### Reconstruct 2-Phase medium field from x's ###
    def gpExpansionExponentialParallel(self, x):
        """
        :param x: The tensor of inputs of the conductivity field !For Many Samples Nx! (2D Tensor - (Nx, dim(x)))
        :param self.gpEigVals: The tensor of the eigenvalues (1D Tensor - (dim(x), ))
        :param self.gpEigVecs: The tensor of the eigenvectors (2D Tensor - (dim(x), dim(IntPoints))
        :return: The Conductivity field for the specific inputs x ( C(x) ) (2D Tensor - (Nx, dim(IntPoints))
        Needs to be transposed for plotting
        """

        out = torch.einsum('i,...i,ij->...j', torch.sqrt(self.gpEigVals),
                                          x, self.gpEigVecs)
        out = torch.reshape(out, (*x.size()[:-1], self.gpgrid.size(dim=1), self.gpgrid.size(dim=1)))
        if len(out.shape) == 4:
            out = torch.nn.functional.interpolate(out, size=(self.sgrid.size(dim=1), self.sgrid.size(dim=2)), mode='bilinear', align_corners=True)
        elif len(out.shape) == 2:
            out = torch.nn.functional.interpolate(out.unsqueeze(0).unsqueeze(0), size=(self.sgrid.size(dim=1), self.sgrid.size(dim=2)), mode='bilinear', align_corners=True)
        mask = out > (self.fraction * torch.std(out, dim=[-1, -2]) + (torch.mean(out, dim=[-1, -2]))).view(-1, 1, 1, 1)
        out = torch.where(mask, self.phaseHigh, self.phaseLow)
        return out








    
    ### Residual Calculation for the ELBO ###
    def calcSingleResGeneralParallel(self, x, y, phi):
   
        c_x = self.gpExpansionExponentialParallel(x)
        
        if self.Linear:
            res = trapzInt2DParallel(
                (- torch.einsum('...ijk,...ijk->...jk',
                                torch.einsum('...ijk,...jk->...ijk', self.shapeFunc.cdWeighFuncParallel(phi), c_x),
                                self.shapeFunc.cdTrialSolutionParallel(y)) \
                - self.rhs * self.shapeFunc.cWeighFuncParallel(phi)))
        else:
            c_x = torch.einsum('...jk,...jk->...jk', c_x, torch.exp((self.shapeFunc.cTrialSolutionParallel(y)-self.u0PDE)*self.alphaPDE))
            res = trapzInt2DParallel(
                (- torch.einsum('...ijk,...ijk->...jk',
                                torch.einsum('...ijk,...jk->...ijk', self.shapeFunc.cdWeighFuncParallel(phi), c_x),
                                self.shapeFunc.cdTrialSolutionParallel(y)) \
                - self.rhs * self.shapeFunc.cWeighFuncParallel(phi)))

        return res


    ### Calculate Empirically the Mean and the Standard Deviation of the Posterior Predictions ###
    def createMeanPredictionsParallel(self, y, yMean):
        yPred = self.shapeFunc.cTrialSolutionParallel(y)
        yMean = self.shapeFunc.cTrialSolutionParallel(yMean)
        yMean = torch.mean(yMean, dim=1)
        ySamples = torch.reshape(yPred, [-1, yPred.size(-2), yPred.size(-1)])
        yStd = torch.std(yPred, dim=1)

        return ySamples, yMean, yStd


    ### Produce Validation Dataset ###
    def produceTestSample(self, Nx=100, solveWithFenics=False, post=None):
        self.shapeFunc.createShapeFuncsConstraint()
        x = torch.zeros(Nx, self.numOfInputs)
        for i in range(0, Nx):
            x[i, :] = torch.randn(self.numOfInputs)
        MeanRelAbsErr = 0.
        NormAbsErr = 0.
        maxMeshLimit = self.options['refSolverIntGrid']
        print("Max Mesh Limit for Fenics: "+ str(maxMeshLimit))
        C_x = torch.zeros(Nx, self.sgrid.size(1), self.sgrid.size(1))
        Sol = torch.zeros(Nx, self.sgrid.size(1), self.sgrid.size(1))
        Res_y = torch.zeros(Nx, self.NofShFuncs)
        SolFenics = torch.zeros(Nx, self.sgrid.size(1), self.sgrid.size(1))
        for i in range(0, x.size(dim=0)):
            # Creating properly the conductivity field from the inputs
            c_x = self.gpExpansionExponentialParallel(torch.reshape(x[i, :], [-1]))
            if len(c_x.shape) > 2:
                c_x = torch.reshape(c_x, [c_x.size(-1), c_x.size(-2)])
            # Solving with Fenics
            solveWithFenics = True
            if solveWithFenics:
                c_xFenics = torch.nn.functional.interpolate(torch.reshape(c_x, [1, 1, c_x.size(0), c_x.size(1)]), size=(maxMeshLimit, maxMeshLimit), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
                solFenics = solve_pde(c_xFenics.cpu().numpy(), self.rhs, uBc=self.uBc, options=self.optionsMod, idx=self.idx).to(device)
                solFenics = torch.nn.functional.interpolate(torch.reshape(solFenics, [1, 1, solFenics.size(0), solFenics.size(1)]),
                                                                size=(self.sgrid.size(-1), self.sgrid.size(-1)), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

            print("Solving with Fenics for i= "+str(i))
            if i < 100:
                # Assemble the system, by using rbfs as basis functions.
                self.shapeFunc.assembleSystem(c_x, self.rhs)
                # Get the coefficients y by soving the system
                res_y = self.shapeFunc.solveNumericalSys()
            else: 
                res_y = torch.zeros_like(Res_y[i-1, :])
            # Obtain the solution in the integration grid.
            sol = torch.reshape(self.shapeFunc.cTrialSolution(res_y), [self.sgrid.size(dim=1), -1])

            # Saving the results
            torch.save(sol, './model/pde/RefSolutions/grid' + str(self.sgrid.size(dim=1)) +
                       'FenicsSample' + str(Nx) + '/sol' + "{:}".format(i) + '.csv')
            torch.save(solFenics, './model/pde/RefSolutions/grid' + str(self.sgrid.size(dim=1)) +
                       'FenicsSample' + str(Nx) + '/solFenics' + "{:}".format(i) + '.csv')
            torch.save(res_y, './model/pde/RefSolutions/grid' + str(self.sgrid.size(dim=1)) +
                       'FenicsSample' + str(Nx) + '/yCoeff' + "{:}".format(i) + '.csv')
            torch.save(c_x, './model/pde/RefSolutions/grid' + str(self.sgrid.size(dim=1)) +
                       'FenicsSample' + str(Nx) + '/Cond' + "{:}".format(i) + '.csv')
            torch.save(x[i, :], './model/pde/RefSolutions/grid' + str(self.sgrid.size(dim=1)) +
                       'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv')
            Sol[i, :, :] = sol
            SolFenics[i, :, :] = solFenics
            C_x[i, :, :] = c_x
            Res_y[i, :] = res_y



            if solveWithFenics:
                MeanRelAbsErr += torch.mean(torch.abs(solFenics - sol) / torch.abs(solFenics + 10 ** (-6)))
                NormAbsErr += torch.linalg.norm(torch.abs(solFenics - sol))
        

        if post is not None:
            sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = post.readTestSample(Nx)
            
        self.shapeFunc.createShapeFuncsFree()
        
        return sampSol, sampSolFenics, sampCond, sampX, sampYCoeff
    
    def saveFields(self, sol, solFenics, yCoeff, cond, x):
        Nx = sol.size(0)
        gridSize = solFenics.size(-1)
        for i in range(0, Nx):
                createFolderIfNotExists('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx))
                torch.save(sol[i, :, :], './model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/sol' + "{:}".format(i) + '.csv')
                torch.save(solFenics[i, :, :], ('./model/pde/RefSolutions/grid' + str(gridSize) +
                                                 'FenicsSample'+str(Nx)+'/solFenics' + "{:}".format(i) + '.csv'))
                torch.save(yCoeff[i, :], './model/pde/RefSolutions/grid' + str(gridSize) +
                                           'FenicsSample' + str(Nx) + '/yCoeff' + "{:}".format(i) + '.csv')
                torch.save(cond[i, :, :], './model/pde/RefSolutions/grid' + str(gridSize) +
                                      'FenicsSample' + str(Nx) + '/Cond' + "{:}".format(i) + '.csv')
                torch.save(x[i, :], './model/pde/RefSolutions/grid' + str(gridSize) +
                                                  'FenicsSample' + str(Nx) + '/x' + "{:}".format(i) + '.csv')
    



