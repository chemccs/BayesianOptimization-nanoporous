import torch
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor





def expected_improvement(X, model, y_best):
    # Get the mean prediction for each point
    mu = model.predict(X)
    
    # Get the standard deviation by looking at the spread of predictions from all estimators (trees)
    predictions = np.array([tree.predict(X) for tree in model.estimators_])
    sigma = np.std(predictions, axis=0)
    
    # Calculate Expected Improvement based on normal distribution
    imp = mu - y_best
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    
    return ei


def upper_confidence_bound(X, model, beta=0.4 ): 
    # Get the mean prediction for each point
    mu = model.predict(X)
    
    # Get the standard deviation by looking at the spread of predictions from all estimators (trees)
    predictions = np.array([tree.predict(X) for tree in model.estimators_])
    sigma = np.std(predictions, axis=0)
    
    # Calculate UCB based on normal distribution
    ucb = mu + beta * sigma
    
    return ucb









def bo_run(nb_iterations, nb_MOFs_initialization):

    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_MOFs_initialization, replace=False)
    
    
    # initialize acquired y, standardize outputs using *only currently acquired data*
    y_acquired = y[ids_acquired].unsqueeze(-1)  # Add an extra dimension
    y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
    y_acquired_1D = y_acquired.squeeze()

    for i in range(nb_MOFs_initialization, nb_iterations):
        
        # construct and fit GP model
        rf = RandomForestRegressor(n_estimators=50, random_state=0)
        #rf =     ExtraTreesRegressor(n_estimators=50, random_state=0)
        model = rf.fit(X[ids_acquired, :], y_acquired_1D)
        
        y_best = y_acquired.max().item()
        X_np = X_unsqueezed.squeeze().cpu().numpy()
        #ei_values  = expected_improvement(X_np, model, y_best)
        ei_values  = upper_confidence_bound(X_np, model, y_best)
        sorted_indices = np.argsort(ei_values)[::-1]

        # Find the first index that is not in ids_acquired
        id_max_ei = next((idx for idx in sorted_indices if idx not in ids_acquired), None)

       
        # acquire this MOF
        ids_acquired = np.concatenate((ids_acquired, [id_max_ei]))
        if(TT[id_max_ei]==50.28):
            break
        
        y_acquired = y[ids_acquired].unsqueeze(-1)
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        y_acquired_1D = y_acquired.squeeze()

        
        

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






