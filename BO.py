import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots as sp

plt.style.use('science')
sns.set(style="darkgrid")



class CustomGPModel(SingleTaskGP):
    """
    Custom Gaussian Process model with a Matern kernel.
    
    Attributes:
        train_x (Tensor): Training input data.
        train_y (Tensor): Training target data.
        likelihood (Likelihood): Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(CustomGPModel, self).__init__(train_x, train_y, likelihood)
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))  # Using Matern kernel here


def bo_run(nb_iterations, nb_MOFs_initialization, store_explore_exploit_terms=False):

    """
    Perform Bayesian Optimization to select MOFs based on their H2 uptake capacity.

    Args:
        nb_iterations: Number of iterations to run the optimization.
        nb_MOFs_initialization: Number of MOFs to initialize with.
        X: Input features tensor.
        y: Target values tensor.
        X_unsqueezed: Input features tensor with an extra dimension.
        TARGET_VALUES: Array of target values for MOFs.
        Gamma_Prior: Flag to use gamma prior in GP model.
        EI: Flag to use Expected Improvement as the acquisition function.
        
    Returns:
        ids_acquired: Indices of acquired MOFs.
    """



    """
    Randomly selects initial MOFs for training data.
    """
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
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # set up acquisition function
        if(EI==1):
           acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
        else:
           beta = 1.0 
           acquisition_function = UpperConfidenceBound(model, beta=beta)

        with torch.no_grad():
            acquisition_values = acquisition_function.forward(X_unsqueezed)
        
        # select MOF to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        id_max_aquisition = next(id.item() for id in ids_sorted_by_aquisition if id.item() not in ids_acquired)

        # acquire this MOF
        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))

        

        # update y aquired; start over to normalize properly
        y_acquired = y[ids_acquired].unsqueeze(-1)  # Add an extra dimension
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)

        
        max_y_idx = y_acquired.argmax().item()
        max_y = TARGET_VALUES[id_max_aquisition]
        if(max_y==50.28):
           break
        
            

    return ids_acquired







vectors = np.loadtxt('MOF_Data.txt', delimiter=',') 


X1 = vectors[:, :7]
TARGET_VALUES = vectors[:, -1]

nb_MOFs = len(vectors)


# Min-Max scaling to the unit cube
X1_min = np.min(X1, axis=0)
X1_max = np.max(X1, axis=0)
X1 = (X1 - X1_min) / (X1_max - X1_min)



# Convert to torch tensors
X = torch.from_numpy(X1)
y = torch.from_numpy(vectors[:, -1])
X_unsqueezed = X.unsqueeze(1)

nb_iterations = 100
nb_runs = 100

EI_W = [0]*nb_iterations #EI with prior
UCB_W = [0]*nb_iterations #UCB with prior

EI_WO = [0]*nb_iterations #EI without prior
UCB_WO = [0]*nb_iterations #UCB without prior


for Gamma_Prior in range(2):
 for EI in range(2):
  BO_Performance = [0]*nb_iterations   
  for r in range(1,nb_runs+1):
    ids_acquired = bo_run(nb_iterations, 5)
    ids_acquired = np.array(ids_acquired)
    maxi =0
    for i in range(len(ids_acquired)):
       tensor_value = (y[ids_acquired[i]])
       extracted_value = tensor_value.item()
       maxi = max(maxi, extracted_value)
       BO_Performance[i] = BO_Performance[i]+maxi
    for i in range(len(ids_acquired),nb_iterations):
       BO_Performance[i] = BO_Performance[i] + maxi
    

  for i in range(nb_iterations):
   if(EI==1 and Gamma_Prior==1): 
      EI_W[i] = BO_Performance[i]/nb_runs
   if(EI==1 and Gamma_Prior==0): 
      EI_WO[i] = BO_Performance[i]/nb_runs 

   if(EI==0 and Gamma_Prior==1): 
      UCB_W[i] = BO_Performance[i]/nb_runs 

   if(EI==0 and Gamma_Prior==0): 
      UCB_WO[i] = BO_Performance[i]/nb_runs 

 
#RD store values for Random Serach
RD = [0]*nb_iterations 

for r in range(1,nb_runs+1):
    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_iterations, replace=False)
    ids_acquired = np.array(ids_acquired)
    maxi = 0
    for i in range(len(ids_acquired)):
       tensor_value = y[ids_acquired[i]]
       extracted_value = tensor_value.item()
       maxi = max(maxi, extracted_value)
       RD[i] = RD[i]+maxi
    for i in range(len(ids_acquired),nb_iterations):
       RD[i] = RD[i] + maxi

for i in range(nb_iterations):
    RD[i] = RD[i]/nb_runs
   



# Create a figure
plt.figure(figsize=(10, 6))

# Plot all five datasets on the same figure
plt.plot(range(1, len(EI_WO) + 1), EI_WO, label='EI (w/o prior)', linestyle='-', marker='o', markersize=4,  color='red')
plt.plot(range(1, len(UCB_WO) + 1), UCB_WO, label='UCB (w/o prior)', linestyle='-', marker='o', markersize=4,  color='blue')

plt.plot(range(1, len(EI_W) + 1), EI_W, label='EI (with prior)', linestyle='-', marker='o', markersize=4, color='green')
plt.plot(range(1, len(UCB_W) + 1), UCB_W, label='UCB (with prior)', linestyle='-', marker='o', markersize=4, color='purple')
plt.plot(range(1, len(RD) + 1), RD, label='Random search', linestyle='-', marker='o', markersize=4, color='orange')

# Add labels and legend
plt.xlabel('No. of evaluated MOFs',fontsize=14, weight='bold')
plt.ylabel('Maximum $H_{2}$ uptake capacity\n among acquired MOFs (wt.\%)',fontsize=14, weight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.legend()

plt.grid(True)
plt.savefig(f"converge_3.png",  dpi=500)
plt.show()


    
