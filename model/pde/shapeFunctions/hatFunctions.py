import torch
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from model.pde.numIntegrators.trapezoid import trapzInt2D, trapzInt2DParallel, simpsonInt2DParallel
import scipy
import time
import warnings
from model.pde.pdeTrueSolFenics import solve_pde
import os
from utils.variousFunctions import memoryOfTensor




class rbfInterpolation:
    def __init__(self, grid, gridW, gridW2, sgrid, tau, tauW, rhs=None, uBc=None, reducedDim=None, savePath=None, options=None):
        """
        :param grid: Center points (in the form of a grid) of each RBF shape function
        :param sgrid: Integration Grid for performing the numerical integration
        :param tau: Scaling parameters for each shape function. Currently this is the same for every shape function.
        :param dl:
        """
        self.gridW = gridW
        self.gridW2 = gridW2
        self.rhs = rhs
        self.uBc = uBc
        self.grid = grid
        self.sgrid = sgrid
        self.tau = tau
        self.tauW = tauW
        self.savePath = savePath


        self.c = 1.
        self.dirFunc = self.ByConstrFuncForDirichlet()
        self.dirFuncGradx()
        self.dirFuncGrady()

        ### Parameters for Non-Linear CG PDE
        self.alphaPDE = 0.
        self.u0PDE = options['u0']

        ### Shape Functions for building the trial functions ####
        self.shapeFuncUnconstraint = torch.vmap(self.rbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                                          torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDx = torch.vmap(self.rbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                  torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.rbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                  torch.reshape(self.grid[1, :, :], [-1]))
        
        ### This should be activated when BC != 0 (Changed on 20.03.24)
        
        self.shapeFunc = torch.vmap(self.rbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                              torch.reshape(self.grid[1, :, :], [-1]))

        self.shapeFuncDx = torch.vmap(self.rbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.rbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        
        
        """
        self.shapeFunc = torch.vmap(self.filtRbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                              torch.reshape(self.grid[1, :, :], [-1]))

        self.shapeFuncDx = torch.vmap(self.filtRbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.filtRbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        """
        
        ### Shape Functions for buiding the weight functions ###
        self.shapeFuncWUnconstraint = torch.vmap(self.rbfW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                                          torch.reshape(self.gridW[1, :, :], [-1]))
        self.shapeFuncWDx = torch.vmap(self.rbfGradxW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                  torch.reshape(self.gridW[1, :, :], [-1]))
        self.shapeFuncWDy = torch.vmap(self.rbfGradyW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                  torch.reshape(self.gridW[1, :, :], [-1]))

        
        self.shapeFuncW = torch.vmap(self.filtRbfW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                              torch.reshape(self.gridW[1, :, :], [-1]))

        self.shapeFuncWDx = torch.vmap(self.filtRbfGradxW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                                     torch.reshape(self.gridW[1, :, :], [-1]))
        self.shapeFuncWDy = torch.vmap(self.filtRbfGradyW)(torch.reshape(self.gridW[0, :, :], [-1]),
                                                     torch.reshape(self.gridW[1, :, :], [-1]))
        
        self.shapeFuncWUnconstraint2 = torch.vmap(self.rbfW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                                          torch.reshape(self.gridW2[1, :, :], [-1]))
        self.shapeFuncWDx2 = torch.vmap(self.rbfGradxW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                  torch.reshape(self.gridW2[1, :, :], [-1]))
        self.shapeFuncWDy2 = torch.vmap(self.rbfGradyW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                  torch.reshape(self.gridW2[1, :, :], [-1]))

        
        self.shapeFuncW2 = torch.vmap(self.filtRbfW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                              torch.reshape(self.gridW2[1, :, :], [-1]))

        self.shapeFuncWDx2 = torch.vmap(self.filtRbfGradxW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                                     torch.reshape(self.gridW2[1, :, :], [-1]))
        self.shapeFuncWDy2 = torch.vmap(self.filtRbfGradyW)(torch.reshape(self.gridW2[0, :, :], [-1]),
                                                     torch.reshape(self.gridW2[1, :, :], [-1]))


        ### Transformation matrix a^-1
        self.aInv = self.inverseBasisStiffness()


        ### Transformation matrix B
        self.reducedDim = reducedDim
        
        self.sgridRx, self.sgridRy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        self.sgridR = torch.stack((self.sgridRx, self.sgridRy), dim=0)
        self.dl = self.sgridR[0][1, 0] - self.sgridR[0][0, 0]

        self.nodeList, self.masks, self.nodesMap, self.bcNodes, self.nonBcNodes, self.kind, self.bind, self.bcNodesOrder = self.get_triangular_nodes(self.sgridR)
        self.rnodeList, self.rmasks, self.rnodesMap, self.rbcNodes, self.rnonBcNodes, self.rkind, self.rbind, self.rbcNodesOrder = self.get_triangular_nodes(self.sgridR, maskMesh=self.sgridR)




        triangShapeFuncs = []
        triangShapeFuncsDx = []
        triangShapeFuncsDy = []
        
        for i in range(0, len(self.nodeList)):
            phi, phiDx, phiDy = self.localTriangularBasis(self.sgrid[0], self.sgrid[1], self.nodeList[i], self.masks[i])
            triangShapeFuncs.append(phi)
            triangShapeFuncsDx.append(phiDx)
            triangShapeFuncsDy.append(phiDy)

        self.triangShapeFuncs = torch.stack(triangShapeFuncs, dim=0)
        self.triangShapeFuncsDx = torch.stack(triangShapeFuncsDx, dim=0)
        self.triangShapeFuncsDy = torch.stack(triangShapeFuncsDy, dim=0)

        # shapefunctions (Nele, 3) -> (CGNodes)
        self.triangShapeFuncsNodes = torch.zeros(self.reducedDim**2, self.sgrid.size(-1), self.sgrid.size(-1))
        self.triangShapeFuncsDxNodes = torch.zeros(self.reducedDim**2, self.sgrid.size(-1), self.sgrid.size(-1))
        self.triangShapeFuncsDyNodes = torch.zeros(self.reducedDim**2, self.sgrid.size(-1), self.sgrid.size(-1))
        for i in range(0, self.nodesMap.size(0)):
            for j in range(0, 3):
                self.triangShapeFuncsNodes[self.nodesMap[i, j]] += self.triangShapeFuncs[i, j, :, :]
                self.triangShapeFuncsDxNodes[self.nodesMap[i, j]] += self.triangShapeFuncsDx[i, j, :, :]
                self.triangShapeFuncsDyNodes[self.nodesMap[i, j]] += self.triangShapeFuncsDy[i, j, :, :]


        rtriangShapeFuncs = []
        rtriangShapeFuncsDx = []
        rtriangShapeFuncsDy = []

        for i in range(0, len(self.nodeList)):
            phi, phiDx, phiDy = self.localTriangularBasis(self.sgridRx, self.sgridRy, self.rnodeList[i], self.rmasks[i])
            rtriangShapeFuncs.append(phi)
            rtriangShapeFuncsDx.append(phiDx)
            rtriangShapeFuncsDy.append(phiDy)

        self.rtriangShapeFuncs = torch.stack(rtriangShapeFuncs, dim=0)
        self.rtriangShapeFuncsDx = torch.stack(rtriangShapeFuncsDx, dim=0)
        self.rtriangShapeFuncsDy = torch.stack(rtriangShapeFuncsDy, dim=0)


        self.B = self.fixedMatrixB()

        self.aInvB = torch.einsum('ij,jk->ik', self.aInv, self.B)


        self.aInvBCO = self.compOrth(self.aInvB)





        ### Testing the Triangular Basis Functions
        xx, yy = torch.meshgrid(torch.linspace(0, 3, self.reducedDim), torch.linspace(0, 3, self.reducedDim), indexing='ij')
        xx = torch.sin(xx)*torch.exp(yy)


            

        ### Testing the yCGtoYFG function



        u = self.fieldNodesToCoeffs(xx)

        

        #ttt = (xxx - u)
        
        c_x = torch.ones(self.reducedDim, self.reducedDim)
        
        self.parallelLocalStiffnessMatrix = torch.func.vmap(self.localStiffnessMatrix)
        self.parallelLocalSourceTerm = torch.func.vmap(self.localSourceTerm)

        self.k = self.parallelLocalStiffnessMatrix(torch.stack(self.rnodeList))
        self.bb = self.parallelLocalSourceTerm(torch.stack(self.nodeList))

        k = self.localStiffnessMatrix(self.nodeList[0])

        #k = self.localSourceTerm(self.nodeList[0])
        k =self.parallelLocalSourceTerm(torch.stack(self.nodeList))
        #K, rhsFromDirichlet = self.assembleStiffMatrix(c_x)
        K = self.assembleStiffMatrix(c_x)

        self.B = self.assembleSourceTerm()
        #self.B = self.B + rhsFromDirichlet


        

        b = self.localSourceTerm
        
        
        #self.coeffFromFit = self.findRbfCoeffs(torch.ones(1, 65, 65))


        if False:
            torch.set_printoptions(profile='full')
            print("Here lies the shape function R.I.P")

            #print(self.shapeFunc[3, :, :])
            print(self.shapeFuncDx[10, :, 4])

            self.plotShapeFunctions()
            self.plotShapeFunctionsGrad(x=True)
            #self.plotShapeFunctionsGrad(x=False)
            test = 1

    def createShapeFuncsFree(self):
        self.shapeFunc = torch.vmap(self.rbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                              torch.reshape(self.grid[1, :, :], [-1]))

        self.shapeFuncDx = torch.vmap(self.rbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.rbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        
    def createShapeFuncsConstraint(self):
        self.shapeFunc = torch.vmap(self.filtRbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                              torch.reshape(self.grid[1, :, :], [-1]))

        self.shapeFuncDx = torch.vmap(self.filtRbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.filtRbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))

    def compOrth(self, A):
        B = torch.from_numpy(scipy.linalg.null_space(A.t().cpu())).to(A.device)

        Z  = torch.cat((A, B), dim=1)
        
        #Itest = (Z.t() @ Z)
        #Iflag = torch.allclose(Itest, torch.eye(Z.size(0)), atol=1e-8)

        sum = 0.
        for i in range(0, A.size(1)):
            for j in range(0, B.size(1)):
                sum += torch.dot(A[:, i], B[:, j])

        if sum > 1e-8:
            raise ValueError("Incorrect Complement orthogonal Matrix!")
        return B

    def padBCs(self, u):
        u = u.view(*u.size()[:-1], self.reducedDim-2, self.reducedDim-2)
        u = torch.nn.functional.pad(u, pad=(1, 1, 1, 1), value=0.)
        u = u.view(*u.size()[:-2], self.reducedDim**2)
        u[...,self.bcNodesOrder] = self.uBc
        u = u.view(*u.size()[:-1], self.reducedDim, self.reducedDim)
        return u
    

    def findRbfCoeffs(self, y=None):
        if y == None:
            phi = None
        else:
            phi = (torch.zeros(y.size(0), self.shapeFunc.size(0)) + torch.rand(y.size(0), self.shapeFunc.size(0))*0.1).requires_grad_(True)
            N=10000
            for i in range(0, N):
                optimizer = torch.optim.Adam(params=[phi], lr=0.001, maximize=False, amsgrad=False)
                optimizer.zero_grad()
                loss = torch.linalg.norm(self.cTrialSolutionParallel(phi)-y)
                #loss = torch.max(torch.abs(self.cTrialSolutionParallel(phi)-y))*0.5+0.5*torch.mean(torch.abs(self.cTrialSolutionParallel(phi)-y))
                print(loss)
                loss.backward()
                optimizer.step()
            print('Max Absolute Error of RBF fitting: ', (self.cTrialSolutionParallel(phi)-y).abs().max())
            print('Mean Absolute Error of RBF fitting: ', (self.cTrialSolutionParallel(phi)-y).abs().mean())
            t = 't'
        return phi.detach()
    

    


    def solveCGPDE(self, c_x, f=None, uBc=None):
        ### Initialization
        f = self.rhs
        if c_x.dim() == 2:
            u = torch.zeros(self.nonBcNodes.size())
        else:
            u = torch.zeros((c_x.size(0), self.nonBcNodes.size(0)))

        
        


        
        for i in range(0, 1000): 
            ### Building the System of Equations

            et = torch.exp((self.padBCs(u)-self.u0PDE)*self.alphaPDE)

            K, KY, rhsFromDirichlet = self.assembleStiffMatrix(c_x, uBc, et, self.padBCs(u))

            dRdYinv = torch.linalg.inv(K+KY)
            B = - self.B.repeat(*c_x.size()[:-2], 1)[..., self.nonBcNodes] * f - rhsFromDirichlet

            ### Solving
            R = torch.einsum('...ij,...j->...i', K, u) - B
            uNew = u + torch.einsum('...ij,...j->...i', dRdYinv, - R)
            u = uNew

            if torch.max(torch.abs(R)) < 10**(-6):
                notConverged = False
                break
            else:
                notConverged = True
        if notConverged:
            print("Newton Rapson for the Non-Linear PDE doesn't converge.")
        u = uNew


        ### Reshaping
        u = self.padBCs(u)
        self.KHHH = K
        self.bHHH = B

        return u
    


    def solveCGPDELinear(self, c_x, f=None, uBc=None):
        f = self.rhs
        K, rhsFromDirichlet = self.assembleStiffMatrix(c_x, uBc)
        Kinv = torch.linalg.inv(K)
        B = - self.B.repeat(*c_x.size()[:-2], 1)[..., self.nonBcNodes] * f - rhsFromDirichlet
        u = torch.einsum('...ij,...j->...i', Kinv, B)
        u = u.view(*u.size()[:-1], self.reducedDim-2, self.reducedDim-2)
        u = torch.nn.functional.pad(u, pad=(1, 1, 1, 1), value=0.)
        u = u.view(*u.size()[:-2], self.reducedDim**2)
        u[...,self.bcNodesOrder] = self.uBc
        u = u.view(*u.size()[:-1], self.reducedDim, self.reducedDim)
        self.KHHH = K
        self.bHHH = B
        #Fenics Test
        #uFenics = solve_pde(c_x[0].clone().detach().cpu(), rhs=-100., uBc=uBc).to(c_x.device)
        #uH = u[0]
        #t = torch.sum(uFenics - u[0])
        return u
    
    def localToGlobalTriang(self, ksi, eta, elemNodes):
        xGlobal = elemNodes[0, 0] + (elemNodes[0, 1] - elemNodes[0, 0]) * ksi + (elemNodes[0, 2] - elemNodes[0, 0]) * eta
        yGlobal = elemNodes[1, 0] + (elemNodes[1, 1] - elemNodes[1, 0]) * ksi + (elemNodes[1, 2] - elemNodes[1, 0]) * eta
        return xGlobal, yGlobal
    
    def get_triangular_nodes(self, mesh, maskMesh=None, zeroBC=False):
        if maskMesh is None:
            maskMesh = self.sgrid
        nodes = []
        masks = []
        bcNodes = []
        bcNodesUp = []
        bcNodesRight = []
        bcNodesLow = []
        bcNodesLeft = []
        x = mesh[0]
        y = mesh[1]
        nodeID = torch.arange(0, x.size(0)*x.size(1), dtype=torch.int32).reshape(x.size(0), x.size(1)).t()
        for j in range(y.size(0) - 1):
            for i in range(x.size(0) - 1):
                # Define triangular elements by connecting adjacent nodes
                uHard1 = torch.tensor([x[i, j], x[i+1, j], x[i, j+1]])
                uHard2 = torch.tensor([x[i, j+1], x[i+1, j], x[i+1, j+1]])
                triangle1 = torch.tensor([[x[i, j], x[i+1, j], x[i+1, j+1]], [y[i, j], y[i+1, j], y[i+1, j+1]], [nodeID[i, j], nodeID[i+1, j], nodeID[i+1, j+1]]])
                triangle2 = torch.tensor([[x[i, j], x[i+1, j+1], x[i, j+1]], [y[i, j], y[i+1, j+1], y[i, j+1]], [nodeID[i, j], nodeID[i+1, j+1], nodeID[i, j+1]]])
                tt = torch.where(maskMesh[0] <= triangle1[0, 1].item(), 1. , 0.) + torch.where(maskMesh[0] >= triangle1[0, 0].item(), 1. , 0.) + \
                torch.where(maskMesh[1] <= triangle1[1, 2].item(), 1. , 0.) + torch.where(maskMesh[1] >= triangle1[1, 0].item(), 1. , 0.)
                mask = torch.where(tt > 3.5, True, False)
                maskk = torch.where(tt > 3.5, True, False).clone()
                Ncut = int(torch.sqrt(torch.sum(mask)))
                mask1 = mask[i*Ncut-i:(i+1)*Ncut-i, j*Ncut-j:(j+1)*Ncut-j]
                mask2 = maskk[i*(Ncut-1):(i+1)*Ncut-i, j*Ncut-j:(j+1)*Ncut-j]
                for k in range(Ncut):
                    for m in range(Ncut):
                        if k < m:
                            mask1[k, m] = False
                        if not j==0:
                            mask1[-1, 0] = False
                        if not i== x.size(0) - 2:
                            mask1[-1, 0] = False
                        if not j == y.size(0) - 2 and not i == x.size(0) - 2:
                            mask1[-1, -1] = False

                mask11 = mask.clone()

                for k in range(Ncut):
                    for m in range(Ncut):
                        if k >= m:
                            mask2[k, m] = False
                        if not j == y.size(0) - 2:
                            mask2[:, -1] = False
                        if not i == 0:
                            mask2[0, :] = False
                            
                        


                mask22 = maskk.clone()

                nodes.append(triangle1)
                nodes.append(triangle2)
                masks.append(mask11)
                masks.append(mask22)
        for j in range(y.size(0)):
            for i in range(x.size(0)):
                if x[i, j] == 0. or x[i, j] == 1. or y[i, j] == 0. or y[i, j] == 1.:
                    bcNodes.append(nodeID[i, j])
                if y[i, j] == 1. and x[i, j] != 1.:
                    bcNodesUp.append(nodeID[i, j])
                if y[i, j] == 0.:
                    bcNodesLow.append(nodeID[i, j])
                if x[i, j] == 0. and y[i, j] != 1. and y[i, j] != 0.:
                    bcNodesLeft.append(nodeID[i, j])
                if x[i, j] == 1. and y[i, j] != 0.:
                    bcNodesRight.append(nodeID[i, j])
        

        nodesMap = torch.stack(nodes)[:, 2, :].to(int)
        bcNodes = torch.stack(bcNodes).t().to(int)
        bcNodesUp = torch.stack(bcNodesUp).t().to(int).tolist()
        bcNodesLow = torch.stack(bcNodesLow).t().to(int).tolist()
        bcNodesRight = torch.stack(bcNodesRight).t().to(int).tolist()
        bcNodesLeft = torch.stack(bcNodesLeft).t().to(int).tolist()
        bcNodesOrdered = [bcNodesLow, bcNodesRight, bcNodesUp[::-1], bcNodesLeft[::-1]]
        flattenedList = [item for sublist in bcNodesOrdered for item in sublist]
        bcNodesOrdered = torch.tensor(flattenedList)
        nonBcNodes = nodeID[1:-1, 1:-1].t().flatten()

        nx = torch.zeros(len(nodes), 3, 3)
        ny = torch.zeros(len(nodes), 3, 3)
        nk = torch.zeros(len(nodes), 3, 3)
        nb = torch.zeros(len(nodes), 3)

        for i in range(0, len(nodes)):
            nx[i], ny[i] = torch.meshgrid(nodesMap[i, :], nodesMap[i, :], indexing='ij')
            nk[i] = torch.ones(3, 3)*i
            nb[i] = torch.ones(3)*i

        #Kp = torch.arange(0, 81).view(9, 9)
        kind = torch.stack((nx, ny, nk), dim=0)
        kind = torch.reshape(kind, [kind.size(0), -1])
        bind = nb.flatten()

        testMasks = torch.stack(masks).to(int)
        CummulMasks = torch.sum(testMasks, dim=0)
        if CummulMasks.max() != 1. or CummulMasks.min() != 1.:
            #warnings.warn("Problem with the triangular shape functions!")
            raise ValueError("Problem with the triangular shape functions!")

        
        return nodes, masks, nodesMap, bcNodes, nonBcNodes, kind.to(int), bind.to(int), bcNodesOrdered
    
    
    def localStiffnessMatrix(self, elemNodes):
        """
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: The local triangular basis functions
        """
        
        JacDeterm = self.dl**2 # = 2*A
        Area = JacDeterm/2.
        phi1Dx = (elemNodes[1, 1] - elemNodes[1, 2])/JacDeterm
        phi2Dx = (elemNodes[1, 2] - elemNodes[1, 0])/JacDeterm
        phi3Dx = (elemNodes[1, 0] - elemNodes[1, 1])/JacDeterm
        phiDx = torch.stack((phi1Dx, phi2Dx, phi3Dx), dim=0)

        phi1Dy = (elemNodes[0, 2] - elemNodes[0, 1])/JacDeterm
        phi2Dy = (elemNodes[0, 0] - elemNodes[0, 2])/JacDeterm
        phi3Dy = (elemNodes[0, 1] - elemNodes[0, 0])/JacDeterm
        phiDy = torch.stack((phi1Dy, phi2Dy, phi3Dy), dim=0)
        phiD = torch.stack((phiDx, phiDy), dim=0)

        K = torch.einsum('ij,ik->jk',phiD, phiD)*Area
        
        return K

    
    def localSourceTerm(self, elemNodes):
        """
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: The local triangular basis functions
        """
        
        Area = self.dl**2 /2./3. # = 2*A
        #Area = self.dl**2
        #phi1 = ((elemNodes[0, 1]*elemNodes[1, 2]) - (elemNodes[0, 2]*elemNodes[1, 1]) + (elemNodes[1, 1] - elemNodes[1, 2])*self.sgridRx + (elemNodes[0, 2] - elemNodes[0, 1])*self.sgridRy)/JacDeterm
        #phi2 = ((elemNodes[0, 2]*elemNodes[1, 0]) - (elemNodes[0, 0]*elemNodes[1, 2]) + (elemNodes[1, 2] - elemNodes[1, 0])*self.sgridRx + (elemNodes[0, 0] - elemNodes[0, 2])*self.sgridRy)/JacDeterm
        #phi3 = ((elemNodes[0, 0]*elemNodes[1, 1]) - (elemNodes[0, 1]*elemNodes[1, 0]) + (elemNodes[1, 0] - elemNodes[1, 1])*self.sgridRx + (elemNodes[0, 1] - elemNodes[0, 0])*self.sgridRy)/JacDeterm
        #phi = torch.stack((phi1, phi2, phi3), dim=0)\
        phi = torch.ones(3)*Area


        return phi

    def localTriangularBasis(self, x, y, elemNodes, mask):
        """
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: The local triangular basis functions
        """
        JacDeterm = self.dl**2
        phi1 = ((elemNodes[0, 1]*elemNodes[1, 2]) - (elemNodes[0, 2]*elemNodes[1, 1]) + (elemNodes[1, 1] - elemNodes[1, 2])*x + (elemNodes[0, 2] - elemNodes[0, 1])*y)/JacDeterm
        phi1 = mask * phi1
        phi2 = ((elemNodes[0, 2]*elemNodes[1, 0]) - (elemNodes[0, 0]*elemNodes[1, 2]) + (elemNodes[1, 2] - elemNodes[1, 0])*x + (elemNodes[0, 0] - elemNodes[0, 2])*y)/JacDeterm
        phi2 = mask * phi2
        phi3 = ((elemNodes[0, 0]*elemNodes[1, 1]) - (elemNodes[0, 1]*elemNodes[1, 0]) + (elemNodes[1, 0] - elemNodes[1, 1])*x + (elemNodes[0, 1] - elemNodes[0, 0])*y)/JacDeterm
        phi3 = mask * phi3
        phi = torch.stack((phi1.t(), phi2.t(), phi3.t()), dim=0)

        phi1Dx = (elemNodes[1, 1] - elemNodes[1, 2])/JacDeterm
        phi1Dx = mask * phi1Dx
        phi2Dx = (elemNodes[1, 2] - elemNodes[1, 0])/JacDeterm
        phi2Dx = mask * phi2Dx
        phi3Dx = (elemNodes[1, 0] - elemNodes[1, 1])/JacDeterm
        phi3Dx = mask * phi3Dx
        phiDx = torch.stack((phi1Dx.t(), phi2Dx.t(), phi3Dx.t()), dim=0)

        phi1Dy = (elemNodes[0, 2] - elemNodes[0, 1])/JacDeterm
        phi1Dy = mask * phi1Dy
        phi2Dy = (elemNodes[0, 0] - elemNodes[0, 2])/JacDeterm
        phi2Dy = mask * phi2Dy
        phi3Dy = (elemNodes[0, 1] - elemNodes[0, 0])/JacDeterm
        phi3Dy = mask * phi3Dy
        phiDy = torch.stack((phi1Dy.t(), phi2Dy.t(), phi3Dy.t()), dim=0)

        return phi, phiDx, phiDy
    
    def fieldNodesToCoeffs(self, u):
        u = u.flatten()
        ucoeff = u[self.nodesMap]
        u = torch.einsum('mk,mkij->ij', ucoeff, self.triangShapeFuncs)
        return u
    
    def assembleStiffMatrix(self, c_x, uBc=None, et=None, ut=None):

        nodesMap = self.rnodesMap
        c_x = c_x.view(*c_x.size()[:-2], -1)
        c_x = c_x[..., self.rnodesMap]
        if et is not None:
            et = et.view(*et.size()[:-2], -1)
            et = et[..., self.rnodesMap]
            em = torch.mean(et, dim=-1).view(*et.size()[:-2], -1, 1).repeat(*(1 for _ in range(et.dim() - 1)), 3)
            ut = ut.view(*ut.size()[:-2], -1)
            ut = ut[..., self.rnodesMap]
        #c_x = torch.mean(c_x, dim=-1).view(-1, 1).repeat(1, 3) # For averaging over all the conductivities of the element
        c_x = torch.mean(c_x, dim=-1).view(*c_x.size()[:-2], -1, 1).repeat(*(1 for _ in range(c_x.dim() - 1)), 3)
        
        if et is not None:
            uele = torch.einsum('...ij,ijk->...ijk', c_x, self.k)*self.alphaPDE/3.
            uele = torch.einsum('...im,...ijk->...imjk', et, uele)
            uele = torch.einsum('...imjk,...ik->...ijm', uele, ut)
            c_x = torch.einsum('...ij,...ij->...ij', c_x, em)




        
        k = torch.einsum('...ij,ijk->...ijk', c_x, self.k)
        Kp = torch.zeros(*c_x.size()[:-2], nodesMap.size(0), self.reducedDim**2, self.reducedDim**2)
        
        
        KpY = torch.zeros(*c_x.size()[:-2], nodesMap.size(0), self.reducedDim**2, self.reducedDim**2)      
        

        #k = torch.reshape(k, [-1])
        k = k.view(*k.size()[:-3], -1)
        
        Kp[..., self.rkind[2], self.rkind[0], self.rkind[1]] = k[..., ] 
        if et is not None:
            uele = uele.view(*uele.size()[:-3], -1)
            KpY[..., self.rkind[2], self.rkind[0], self.rkind[1]] = uele[..., ]
            KpY = torch.sum(KpY, dim=-3)
            KtY = KpY[..., self.nonBcNodes, :]
            KtY = KtY[..., :, self.nonBcNodes]
        #KpY[..., self.rkind[2], self.rkind[4], self.rkind[0], self.rkind[1]] = uele[..., ] 
        Kp = torch.sum(Kp, dim=-3)
        

        Kt = Kp[..., self.nonBcNodes, :]
        

        ### For boundary conditions that are equal to 0
        if uBc is None:
            self.uBc = torch.zeros(self.bcNodes.size(0))
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodes], self.uBc)
        elif isinstance(uBc, float):
            self.uBc = torch.ones(self.bcNodes.size(0)) * uBc
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodes], self.uBc)
        elif uBc.dim() == 1:
            self.uBc = uBc
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodesOrder], self.uBc)

        
        Kt = Kt[..., :, self.nonBcNodes]
        

        ### For boundary conditions that arent equal to 0

        if et is not None:
            return Kt, KtY, rhsFromBCsDiriclet
        else:
            KtY = torch.zeros(1)
            return Kt, KtY, rhsFromBCsDiriclet
    

    def assembleStiffMatrixOrig(self, c_x, uBc=None, et=None):
        
        nodesMap = self.rnodesMap
        c_x = c_x.view(*c_x.size()[:-2], -1)
        c_x = c_x[..., self.rnodesMap]
        if et is not None:
            et = et.view(*et.size()[:-2], -1)
            et = et[..., self.rnodesMap]
        #c_x = torch.mean(c_x, dim=-1).view(-1, 1).repeat(1, 3) # For averaging over all the conductivities of the element
        c_x = torch.mean(c_x, dim=-1).view(*c_x.size()[:-2], -1, 1).repeat(*(1 for _ in range(c_x.dim() - 1)), 3)
        k = torch.einsum('...ij,ijk->...ijk', c_x, self.k)
        Kp = torch.zeros(*c_x.size()[:-2], nodesMap.size(0), self.reducedDim**2, self.reducedDim**2)
             

        #k = torch.reshape(k, [-1])
        k = k.view(*k.size()[:-3], -1)
        
        Kp[..., self.rkind[2], self.rkind[0], self.rkind[1]] = k[..., ] 
        Kp = torch.sum(Kp, dim=-3)

        Kt = Kp[..., self.nonBcNodes, :]

        ### For boundary conditions that are equal to 0
        if uBc is None:
            self.uBc = torch.zeros(self.bcNodes.size(0))
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodes], self.uBc)
        elif isinstance(uBc, float):
            self.uBc = torch.ones(self.bcNodes.size(0)) * uBc
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodes], self.uBc)
        elif uBc.dim() == 1:
            self.uBc = uBc
            rhsFromBCsDiriclet = torch.einsum('...ij,j->...i', Kt[..., :, self.bcNodesOrder], self.uBc)

        
        Kt = Kt[..., :, self.nonBcNodes]

        ### For boundary conditions that arent equal to 0

        return Kt
    
    def assembleSourceTerm(self):

        Bp = torch.zeros(self.rnodesMap.size(0), self.reducedDim**2)
        index = self.rnodesMap.flatten()
        Bp[self.rbind, index] = self.bb.flatten()
        Bp = torch.sum(Bp, dim=0)

        return Bp

    def einsumInBatches(self, tensor1, tensor2, chunk_size):
        out1 = torch.empty(tensor1.size(0), tensor1.size(1), tensor1.size(2), tensor1.size(3))
        for start in range(0, tensor1.size(1), chunk_size):
            end = min(start + chunk_size, tensor1.size(1))

            # Slice the tensors into chunks along the second dimension
            intermediate_chunk = tensor2[:, start:end]
            du_jds_chunk = tensor1[:, start:end]

            # Calculate the tensor product for the chunk and add it to the final result
            out1 = torch.einsum('ijkl,imkl->jmkl', intermediate_chunk, du_jds_chunk)
            out1 = trapzInt2DParallel(out1)
            return


    def assembleSystem(self, c_x, f):
        """
        :param c_x: The conductivity field that needs to be multiplied elementwise with the outer product
        dus_i/ds * dus_j/ds
        :return: This returns the complete matrix A and vector b for solving numerically the system of the weak
        form.
        """
        # Stacking the space derivatives together
        du_ids = torch.stack((self.shapeFuncDx, self.shapeFuncDy), dim=0)
        # Introducing the second derivative
        du_jds = torch.stack((self.shapeFuncDx, self.shapeFuncDy), dim=0)
        # Dot product and outer product in the same line of code.
        intermediate = torch.einsum('ijkl,kl->ijkl', du_ids, c_x)
        if du_jds.size(1) < 300:
            out1 = torch.einsum('ijkl,imkl->jmkl', intermediate, du_jds)
            self.A = trapzInt2DParallel(out1)
        else:
            AAA = torch.zeros(intermediate.size(1), intermediate.size(1))
            if False:
                divider = 20
                batchSize = intermediate.size(1) // divider
                modulo = intermediate.size(1) % divider
                for i in range(0, divider):
                    tempOut = torch.einsum('ijkl,imkl->jmkl', intermediate[:, int(i*batchSize):int((i+1)*batchSize), :, :], du_jds)
                    AAA[int(i*batchSize):int((i+1)*batchSize), :] = trapzInt2DParallel(tempOut)
                if modulo > 0:
                    tempOut = torch.einsum('ijkl,imkl->jmkl',
                                           intermediate[:, int((i + 1) * batchSize):, :, :], du_jds)
                    AAA[int((i + 1) * batchSize):, :] = trapzInt2DParallel(tempOut)
            else:
                for i in range(0, intermediate.size(1)):
                    tempOut = torch.einsum('ikl,imkl->mkl', intermediate[:, i, :, :], du_jds)
                    AAA[i, :] = trapzInt2DParallel(tempOut)
            self.A = AAA


        self.b = - f * trapzInt2DParallel(self.shapeFunc)
        return

    def solveNumericalSys(self):
        self.numSolution = torch.linalg.solve(self.A, self.b)
        return self.numSolution


    def NumericalLaplacian2D(self, z):
        c = self.sgrid.size(dim=1) - 1
        fxx = torch.gradient(torch.gradient(z, dim=0)[0] * c, dim=0)[0] * c
        fyy = torch.gradient(torch.gradient(z, dim=1)[0] * c, dim=1)[0] * c
        return fxx + fyy
    def NumericalGrad1D(self):
        x = torch.linspace(0, 1, 101)
        y = x**2
        dy = 2*x
        disy = torch.gradient(torch.gradient(y)[0]*100.)[0]*100
        err = dy - disy
        return y
    def ByConstrFuncForDirichlet(self):
        """
        It Provides the function with which the weighting function will be multiplied in order to make sure that w(s)=0
        on the boundaries when diriclet conditions are used.
        ATTENTION!: The normalization constant (0.0625) could have a great stabilization effect on the convergence of
        the algorithm, so use it with caution.
        :return:
        """
        return (1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c 
    def dirFuncGradx(self):
        return -self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c + \
               (1-self.sgrid[0, :, :])*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c
    def dirFuncGrady(self):
        return -(1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*self.sgrid[1, :, :]/self.c +\
               (1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])/self.c
    def rbf(self, xnode, ynode):
        return torch.exp(-self.tau*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))
    def rbfGradx(self, xnode, ynode):
        return -2 * self.tau * (self.sgrid[0, :, :] - xnode) * \
             torch.exp(-self.tau * ((xnode - self.sgrid[0, :, :]) ** 2 + (ynode - self.sgrid[1, :, :]) ** 2))
    def rbfGrady(self, xnode, ynode):
        return -2 * self.tau * (self.sgrid[1, :, :] - ynode) *\
             torch.exp(-self.tau*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))

    def filtRbf(self, xnode, ynode):
        return torch.mul(self.rbf(xnode, ynode), self.dirFunc) #+ torch.tensor(self.uBc)
    def filtRbfGradx(self, xnode, ynode):
        return torch.mul(self.rbfGradx(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbf(xnode, ynode), self.dirFuncGradx())
    def filtRbfGrady(self, xnode, ynode):
        return torch.mul(self.rbfGrady(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbf(xnode, ynode), self.dirFuncGrady())
    

    def rbfW(self, xnode, ynode):
        return torch.exp(-self.tauW*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))
    def rbfGradxW(self, xnode, ynode):
        return -2 * self.tauW * (self.sgrid[0, :, :] - xnode) * \
             torch.exp(-self.tauW * ((xnode - self.sgrid[0, :, :]) ** 2 + (ynode - self.sgrid[1, :, :]) ** 2))
    def rbfGradyW(self, xnode, ynode):
        return -2 * self.tauW * (self.sgrid[1, :, :] - ynode) *\
             torch.exp(-self.tauW*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))

    def filtRbfW(self, xnode, ynode):
        return torch.mul(self.rbfW(xnode, ynode), self.dirFunc) #+ torch.tensor(self.uBc)
    def filtRbfGradxW(self, xnode, ynode):
        return torch.mul(self.rbfGradxW(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbfW(xnode, ynode), self.dirFuncGradx())
    def filtRbfGradyW(self, xnode, ynode):
        return torch.mul(self.rbfGradyW(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbfW(xnode, ynode), self.dirFuncGrady())


    def cWeighFunc(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)
    
    
    def inverseBasisStiffness(self):
        """
        :return: The inverse of the stiffness matrix of the basis functions. int_{s} phi_j*phi_k ds
        """
        
        if os.path.exists(self.savePath+ 'AFixedTransformationMatrix.dat'):
            a = torch.load(self.savePath+ 'AFixedTransformationMatrix.dat')
        
        else:
            tt = torch.zeros(self.grid.size(1)**2, self.grid.size(1)**2)
            for i in range(self.grid.size(1)**2):
                    tt[i, :] = trapzInt2DParallel(torch.einsum('ij, kij->kij', self.shapeFunc[i], self.shapeFunc[:]))
            a = tt
            torch.save(a, self.savePath + 'AFixedTransformationMatrix.dat')
        aInv = torch.inverse(a)
        return aInv
    
    def fixedMatrixB(self):
        """
        :return: The inverse of the stiffness matrix of the basis functions. int_{s} phi_j*phi_k ds
        """
        tt = torch.zeros(self.grid.size(1)**2, self.reducedDim**2)
        for i in range(self.grid.size(1)**2):
                tt[i, :] = trapzInt2DParallel(torch.einsum('ij, kij->kij', self.shapeFunc[i], self.triangShapeFuncsNodes[:]))
        b = tt
        return b
    def fixedMatrixBF(self):
        """
        :return: The inverse of the stiffness matrix of the basis functions. int_{s} phi_j*phi_k ds
        """
        tt = torch.zeros(self.grid.size(1)**2, self.reducedDimFull**2) 
        for i in range(self.grid.size(1)**2):
                tt[i, :] = trapzInt2DParallel(torch.einsum('ij, kij->kij', self.shapeFunc[i], self.triangShapeFuncsNodesF[:]))
        b = tt
        return b
    
    def yCGtoYFG(self, yCG):

        if self.reducedDim != yCG.size(-1):
            raise ValueError("The dimension "+  str(yCG.size(-1))+" of the input vector is not consistent with the reduced dimension reducedDim =" + str(self.reducedDim)+" !")
        
        if self.reducedDim == yCG.size(-1):
            yCG = yCG.view(*yCG.size()[:-2], -1) 
            y = torch.einsum('...i,ki->...k', yCG, self.aInvB)

        return y
    
    
    
    def cWeighFuncParallel(self, phi):
        """
        :param phi: Coefficients of the expansion for Nx different samples (NOT RELATED to the grid
         and WITHOUT any physical meaning). This should
        be a 2D tensor consisting of (Nx, dim(phi)) entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        
        return torch.einsum('...i,ijk->...jk', phi, self.shapeFuncW) #+ self.uBc[0]
    
    def cWeighFuncParallel2(self, phi):
        
        return torch.einsum('...i,ijk->...jk', phi, self.shapeFuncW2) #+ self.uBc[0]
    
    def cTrialSolutionParallelTriang(self, phi):

        phi = torch.nn.functional.interpolate(phi.view(phi.size(0), -1, self.reducedDim, self.reducedDim), size=(31, 31), mode='bilinear', align_corners=True)
        t1 = phi.view(phi.size(0), -1, 31, 31)

        return t1
    def cTrialSolutionParallelExternal(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        phi = phi.view(*phi.size()[:-1], self.reducedDim, self.reducedDim)
        phi = self.yCGtoYFG(phi)
        t1 = torch.einsum('...i,ijk->...jk', phi, self.shapeFunc).view(-1, 1, self.shapeFunc.size(-1), self.shapeFunc.size(-1))

        return torch.einsum('...i,ijk->...jk', phi, self.shapeFunc)
    def cTrialSolutionParallel(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('...i,ijk->...jk', phi, self.shapeFunc)
    def cTrialSolution(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)

    def cConductivityFunc(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFuncUnconstraint)
    def cWeighFuncOrig(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)
    def cdWeighFunc(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        dx = torch.einsum('i,ijk->jk', phi, self.shapeFuncDx)
        dy = torch.einsum('i,ijk->jk', phi, self.shapeFuncDy)
        return torch.stack((dx, dy), dim=0)

    def cdWeighFuncParallel(self, phi):
        """
        :param phi: Coefficients of the expansion for Nx different samples (NOT RELATED to the grid and
        WITHOUT any physical meaning). This should
        be a 2D tensor consisting of (Nx, dim(phi)) entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        dx = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncWDx)
        dy = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncWDy)
        return torch.stack((dx, dy), dim=-3)
    
    def cdWeighFuncParallel2(self, phi):
        
        dx = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncWDx2)
        dy = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncWDy2)
        return torch.stack((dx, dy), dim=-3)
    
    def cdTrialSolutionParallel(self, phi, FiniteDifferences=False):
        """
        :param phi: Coefficients of the expansion for Nx different samples (NOT RELATED to the grid and
        WITHOUT any physical meaning). This should
        be a 2D tensor consisting of (Nx, dim(phi)) entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """

        if FiniteDifferences:
            N = 1000
            ### Trial Solution in the grid
            """
            phi = phi.view(*phi.size()[:-1], self.reducedDim, self.reducedDim)
            phi = self.yCGtoYFG(phi)
            t1 = torch.einsum('...i,ijk->...jk', phi, self.shapeFunc).view(-1, 1, self.shapeFunc.size(-1), self.shapeFunc.size(-1))
            """

            
            phi = torch.nn.functional.interpolate(phi.view(*phi.size()[:-1], self.reducedDim, self.reducedDim), size=(31, 31), mode='bilinear', align_corners=True).view(-1, 1, 31**2)
            t1 = phi.view(*phi.size()[:-1], 31, 31)
            #t1 = phi.view(*phi.size()[:-1], self.grid.size(-1), self.grid.size(-1))
            t1 = torch.nn.functional.interpolate(t1, size=(self.sgrid.size(-1), self.sgrid.size(-1)), mode='bilinear', align_corners=True)
            

            ### FDs for x
            phiX = torch.nn.functional.interpolate(t1, size=(N+1, t1.size(-1)), mode='bilinear', align_corners=True).view(-1, N+1, t1.size(-1))
            dyx = torch.diff(phiX*N, dim=-2).unsqueeze(1)
            dyx = torch.nn.functional.interpolate(dyx, size=(t1.size(-2), t1.size(-1)), mode='bilinear', align_corners=True)

            ### FDs for y
            phiY = torch.nn.functional.interpolate(t1, size=(t1.size(-2), N+1), mode='bilinear', align_corners=True).view(-1, t1.size(-2), N+1)
            dyy = torch.diff(phiY*N, dim=-1).unsqueeze(1)
            dyy = torch.nn.functional.interpolate(dyy, size=(t1.size(-2), t1.size(-1)), mode='bilinear', align_corners=True)

            return torch.stack((dyx, dyy), dim=-3)
                

        else:        
            dx = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncDx)
            #dt1 = dx[0, 0, 17:24, 17:24]
            dy = torch.einsum('...i,ijk->...jk', phi, self.shapeFuncDy)
            return torch.stack((dx, dy), dim=-3)
        

    def plotShapeFunctions(self):
        numOfShFuncs = self.grid.size(dim=1)
        gridSize = self.sgrid.size(dim=1)
        fig, axs = plt.subplots(numOfShFuncs, numOfShFuncs, figsize=(12, 12))
        temp = self.shapeFunc.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        for i in range(numOfShFuncs):
            for j in range(numOfShFuncs):
                im = axs[i, j].pcolormesh(self.sgrid[0, :, :].detach().cpu(), self.sgrid[1, :, :].detach().cpu(), temp[i, j, :, :].detach().cpu().T,
                                  cmap='coolwarm', shading='auto')
                axs[i, j].set_title("Shape Function: " + str(i*4+j))
                axs[i, j].set_aspect('equal')
                cbar = fig.colorbar(im, ax=axs[i, j])


        plt.show()
        plt.savefig('./results/figs/ChebyShapeFuncs.png', dpi=300, bbox_inches='tight')
        plt.close('all')


    def plotShapeFunctionsGrad(self, x=True):
        numOfShFuncs = self.grid.size(dim=1)
        gridSize = self.sgrid.size(dim=1)
        fig, axs = plt.subplots(numOfShFuncs, numOfShFuncs, figsize=(12, 12))
        if x == True:
            temp = self.shapeFuncDx.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        else:
            temp = self.shapeFuncDy.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        for i in range(numOfShFuncs):
            for j in range(numOfShFuncs):
                axs[i, j].pcolormesh(self.sgrid[0, :, :].detach().cpu(), self.sgrid[1, :, :].detach().cpu(), temp[i, j, :, :].detach().cpu().T,
                                  cmap='coolwarm', shading='auto')
                axs[i, j].set_title("Shape Function: " + str(i*4+j))
                axs[i, j].set_aspect('equal')

        plt.show()
        plt.savefig('./results/figs/ChebyShapeFuncsGradx.png', dpi=300, bbox_inches='tight')
        plt.close('all')

