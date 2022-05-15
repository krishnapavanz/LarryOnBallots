'''
Data preprocessing functions to identify colinearity among columns
and create dummies for 
'''

import numpy as np
import pandas as pd
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed

# Code taken from  https://stackoverflow.com
def calculate_vif_(X, thresh=5.0):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

#X = df[feature_list] # Selecting your data

#X2 = calculate_vif_(X,5) # Actually running the function

def add_canton_dummies(dataframe):
    '''
    Adds 26 columns to the dataframe to create at dummy variable (1) for each state.
    '''
    for row in dataframe.iterrows():
        if dataframe['canton'] == 'Zurich':
            df['Zurich_Dummy'] = 1
        else:
            df['Zurich_Dummy'] = 0 
    # Need to update with the 25 other cantons.
    return dataframe

