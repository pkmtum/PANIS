import torch
import torch.nn as nn
import torch.nn as nn
from model.pde.numIntegrators.trapezoid import trapzInt2D, trapzInt2DParallel
import matplotlib.pyplot as plt


### Normal CNN
class nnmPANIS(nn.Module): # mPANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnmPANIS, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.


        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 32x32
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)  # 16x16
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act3 = nn.Softplus()
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act4 = nn.Softplus()
        self.bn4 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)

        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
   
    
    def xtoXCnn(self, x, printFlag=False):

        x = self.pde.gpExpansionExponentialParallel(x)
        x = torch.log(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.deconv1(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)

        x = torch.exp(x)


        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


        x = self.Ytoy(x)    


        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
    
    def forwardMultiple(self, x, Navg):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        x = x.repeat(1, Navg, 1).view(10*Navg, 1, -1)
        x = x.view(-1, self.reducedDim**2) +\
              (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                  (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2))) * self.yFMode
        

        x = self.Ytoy(x)    

        x = x.view(-1, Navg, self.pde.NofShFuncs)

        return x
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])

        x = self.pde.shapeFunc.yCGtoYFG(x)

        

        return x
    
    
  
    def XtoY(self, x):
        x = self.pde.shapeFunc.solveCGPDE(c_x=x, f=100., uBc=self.uBc)
        return x
    





class nnPANIS(nn.Module): # PANIS
    def __init__(self, reducedDim, cgCnn, xtoXCnn, pde, extraParams):
        super(nnPANIS, self).__init__()
        self.reducedDim = reducedDim
        self.pde = pde
        self.cgCnn = cgCnn
        self.xtoXnn = xtoXCnn
        self.tess = 0.
        self.uBc = pde.uBc
        self.V = extraParams[0]
        self.globalSigma = extraParams[1]
        self.iter = 0
        if extraParams[2]:
            self.yFMode = 1.
        else:
            self.yFMode = 0.

        
        # x -> X
        
        fb= 8
        ef1 = 3
        ef2 = 2
        fbs= 8
        ef1s = 2
        ef2s = 2
        

        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # 32x32 k3s2p1 (For fast runs k3s1p1)
        self.act1 = nn.Softplus()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  #### k2s2 for reduceDim=17, k4s4 for reduceDim=9 for intGrid129
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1)  # 16x16
        self.act2 = nn.Softplus()
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16
 
        self.deconv2 = nn.ConvTranspose2d(24, 8, kernel_size=4, stride=1, padding=1)  # 32x32
        self.act5 = nn.Softplus()
        self.bn5 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)  # 32x32

        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        
        
        numOfPars = self.count_parameters()
        print("Number of NN Parameters: ", numOfPars)

    
    
   
    
    def xtoXCnn(self, x, printFlag=False):
        x = self.pde.gpExpansionExponentialParallel(x)

        x = torch.log(x)

        
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.deconv2(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.deconv3(x)

        x = torch.exp(x)



        

        x = torch.reshape(x, [x.size(0), self.reducedDim, self.reducedDim])
        return x
        

    def forward(self, x):
        
        x = self.xtoXCnn(x)

        x = self.pde.shapeFunc.solveCGPDE(c_x=x, f=100., uBc=self.uBc)


        ### For reduced covariance
        if self.yFMode:
            x = x.view(-1, self.reducedDim**2) +\
                (torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(x.size(0), self.V.size(-1))) +\
                    (torch.pow(10, self.globalSigma/2.) * torch.randn(x.size(0), self.reducedDim**2)))


        x = self.Ytoy(x)    

        x = x.view(-1, 1, self.pde.NofShFuncs)

        return x
 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def Ytoy(self, x):
        x = torch.reshape(x, [x.size(0), -1, self.reducedDim, self.reducedDim])


        x = self.pde.shapeFunc.yCGtoYFG(x)
        

        return x
    
  
    def XtoY(self, x):
        x = self.pde.shapeFunc.solveCGPDE(c_x=x, f=100., uBc=self.uBc)
        return x


