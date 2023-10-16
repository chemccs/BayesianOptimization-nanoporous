import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import MaternKernel
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
#from scipy.stats import norm


Gamma_Prior = 0


class CustomGPModel(SingleTaskGP):
    def __init__(self, train_x, train_y, likelihood):
        super(CustomGPModel, self).__init__(train_x, train_y, likelihood)
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))  # Using Matern kernel here


def bo_run(nb_iterations, nb_MOFs_initialization, store_explore_exploit_terms=False):

    # select initial MOFs for training data randomly.
    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_MOFs_initialization, replace=False)
    
    # keep track of exploration vs. exploitation terms for EI  
    explore_exploit_balance = np.array([(np.NaN, np.NaN) for i in range(nb_iterations)]) if store_explore_exploit_terms else []

    # initialize acquired y, standardize outputs using *only currently acquired data*
    y_acquired = y[ids_acquired].unsqueeze(-1)  # Add an extra dimension
    y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)

    

    for i in range(nb_MOFs_initialization, nb_iterations):
        # construct and fit GP model
        if(Gamma_Prior == 1):
           model = SingleTaskGP(X[ids_acquired, :], y_acquired)
        else:                   
           likelihood = gpytorch.likelihoods.GaussianLikelihood()
           model = CustomGPModel(X[ids_acquired, :], y_acquired, likelihood)
        #print(model)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # set up acquisition function
        #acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
        beta = 1.0 #0.4
        acquisition_function = UpperConfidenceBound(model, beta=beta)

        with torch.no_grad():
            acquisition_values = acquisition_function.forward(X_unsqueezed)
        
        #max_acquisition_values.append(acquisition_values.max().item())
        # select MOF to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        id_max_aquisition = next(id.item() for id in ids_sorted_by_aquisition if id.item() not in ids_acquired)

        # acquire this MOF
        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))

        

        # update y aquired; start over to normalize properly
        y_acquired = y[ids_acquired].unsqueeze(-1)  # Add an extra dimension
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)

        
        max_y_idx = y_acquired.argmax().item()
        max_y = TT[id_max_aquisition]
        if(max_y==50.28):
           break
        
            

    return ids_acquired







vectors1 = np.loadtxt('MOF_Data.txt', delimiter=',') 


X1 = vectors1[:, :7]

nb_MOFs = len(vectors1)
# Min-Max scaling to the unit cube
X1_min = np.min(X1, axis=0)
X1_max = np.max(X1, axis=0)
X1 = (X1 - X1_min) / (X1_max - X1_min)
TT = vectors1[:, -1]
X = torch.from_numpy(X1)
y = torch.from_numpy(vectors1[:, -1])
X_unsqueezed = X.unsqueeze(1)

nb_iterations = 180
nb_runs = 100

VAL = 0
   
for r in range(1,nb_runs+1):
    ids_acquired=bo_run(nb_iterations, 5)
    B = np.array(ids_acquired)
    C=[]
    for i in range(len(B)):
       tensor_value= (y[B[i]])
       extracted_value = tensor_value.item()
       if(extracted_value==50.28):
           print('1st max hit', i)
           C.append(extracted_value)
           break
       C.append(extracted_value)
    VAL = VAL + i + 1
    print(f"Run {r}")
    print('Average:', round(VAL/(r),10))
    print('-----------------------------------')
























