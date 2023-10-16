import torch
import numpy as np
from scipy.stats import norm
import xgboost as xgb



def expected_improvement(X, model, y_best):
    # Get the mean prediction for each point
    mu = model.predict(X)
    
    # Get the predictions from all individual trees
    n_trees = model.get_booster().best_ntree_limit
    predictions = np.array([model.predict(X, iteration_range=(0, i)) for i in range(1, n_trees + 1)])
    sigma = np.std(predictions, axis=0)
    
    # Calculate Expected Improvement based on normal distribution
    imp = mu - y_best
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    
    return ei

def upper_confidence_bound(X, model, beta=0.4 ): #kappa=2.0
    # Get the mean prediction for each point
    mu = model.predict(X)
    
    # Get the predictions from all individual trees
    n_trees = model.get_booster().best_ntree_limit
    predictions = np.array([model.predict(X, iteration_range=(0, i)) for i in range(1, n_trees + 1)])
    sigma = np.std(predictions, axis=0)
    
    # Calculate UCB based on normal distribution
    ucb = mu + beta * sigma
    
    return ucb















def bo_run(nb_iterations, nb_MOFs_initialization):

    # select initial MOFs for training data randomly.
    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_MOFs_initialization, replace=False)
    
    # initialize acquired y, standardize outputs using *only currently acquired data*
    y_acquired = y[ids_acquired].unsqueeze(-1)  # Add an extra dimension
    y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
    y_acquired_1D = y_acquired.squeeze()

    for i in range(nb_MOFs_initialization, nb_iterations):
        # Construct and fit the XGBRegressor model
        model = xgb.XGBRegressor(
            colsample_bytree=1,
            learning_rate=0.2,
            max_depth=5,
            n_estimators=300,
            subsample=1,
            objective='reg:squarederror',
            random_state=0
        )
        model.fit(X[ids_acquired, :].cpu().numpy(), y_acquired_1D.cpu().numpy()) # Convert to numpy as XGBoost requires numpy arrays
        
        # Set up the acquisition function
        y_best = y_acquired.max().item()
        X_np = X_unsqueezed.squeeze().cpu().numpy()
        ei_values = expected_improvement(X_np, model, y_best)
        #ei_values = upper_confidence_bound(X_np, model, y_best)
        sorted_indices = np.argsort(ei_values)[::-1]
        id_max_ei = next((idx for idx in sorted_indices if idx not in ids_acquired), None)

        # Acquire the sample
        ids_acquired = np.concatenate((ids_acquired, [id_max_ei]))
        if(TT[id_max_ei] == 50.28):
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
    VAL = VAL + i +1
    print(f"Run {r}")
    print('Average:', round(VAL/(r),10))
    print('-----------------------------------')



