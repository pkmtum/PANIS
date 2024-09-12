
cudaIndex = 0 # Index of the GPU to use
device='cuda' # 'cpu' or 'cuda'
dataType = 'double' # 'float' or 'double'
nele = 63 # number of rbfs -1 in each dimension for \eta_i(s)
shapeFuncsDim = 16 # number of rbfs in each dimension for \w_j(s)
reducedDim = 17 # \sqrt{dimX}
createNewCondField = False # True if you want to create a new conductivity field dataset
saveDatasetOption = False # True if you want to save the dataset at the location below
importDatasetOption = True # True if you want to import the dataset from the location below
saveModelOption = False # True if you want to save the model at ./utils/trainedNNs/
outOfDistributionPrediction = False # True if you want to use model from ./utils/trainNNs/ (specified in main.py)
numOfTestSamples = 100 # number of test samples for validation
options={'modeType': 'test' + str(numOfTestSamples), # It is affects only the name of the dataset produced
  'volumeFraction': 'FR50',
  'lengthScale': 0.25,
  'inputDimensions': 1024,
  'boundaryCondition': 0., #float or string not integer
  'alpha': 0.05,
  'u0': 5.,
  'integrationGrid': 129,
  'contrastRatio': 'CR10',
  'volumeFractionOutOfDistribution': 'FR50',
  'refSolverIntGrid': 240}
compareWithCGProjection = False # True if you want to compare with CG projection of the solution, false if you want to compare with ref solution
yFMode=False # True for mPANIS and False for PANIS
Nx_samp = 10 # number of Monte Carlo Samples per iteration
randResBatchSize = 200 # number of randomly selected weight functions to approximate the ELBO
Navg = 100 # Number of samples used for statistics during validation
mean_px = 0 # Prior mean for px
sigma_px = 1 # Prior std for px
IterSvi = 10000 # Number of SVI iterations
lr = 0.01 # Learning rate
sigma_rExponent = 3.5 # In paper it was 3.5 for PANIS and 2.2 for mPANIS
stdInit = 3 # Initiallization of the exponent of the covariance matrix values
sigma_r = 10 ** (-sigma_rExponent)
sigma_w = 10**8 # std for the prior py
powerIterTol = 10 ** (-5) # Tolerance for power iteration when calculating eigenvalues and eigenvectors numerically
display_plots = False
rhs = -100. # Numerical value for the force term in the PDE

# Path of the dataset
saveFolder = './Datasets/CMAMEn005PANIS/darcy_'
if isinstance(options['boundaryCondition'], str):
  saveDatasetName = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+'RefIntGrid'+str(options['refSolverIntGrid']) + \
  options['contrastRatio']+options['volumeFraction']+'D'+options['boundaryCondition']+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
  saveDatasetNameOutOfDist = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+ 'RefIntGrid'+str(options['refSolverIntGrid']) +\
options['contrastRatio']+options['volumeFractionOutOfDistribution']+'D'+options['boundaryCondition']+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
else:
  saveDatasetName = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+'RefIntGrid'+str(options['refSolverIntGrid']) +\
  options['contrastRatio']+options['volumeFraction']+'D'+str(int(options['boundaryCondition']))+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
  saveDatasetNameOutOfDist = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions']) +'RefIntGrid'+str(options['refSolverIntGrid']) +\
options['contrastRatio']+options['volumeFractionOutOfDistribution']+'D'+str(int(options['boundaryCondition']))+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
