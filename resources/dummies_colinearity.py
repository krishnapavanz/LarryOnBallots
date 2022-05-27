'''
Data preprocessing functions to identify colinearity among independent
variables in the 'demographics_2021' dataset and create dummies for each 
canton in the the 'face_covering' dataset.  

calculate_vif function code taken and modified from Stackoverflow
https://stackoverflow.com/questions/63795551/variance-inflation-factor-output-statsmodels
'''
import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(dataframe):
    '''
    This function is written to be used with the 'demographics_2021' dataset.  
    '''
    variables = list(dataframe.columns)
    del variables[0] # Removes the municipality column from the dataframe
    variables = dataframe[variables] # Removes the municipality column from the dataframe
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
    vif['features'] = variables.columns
    return vif


def add_dummies(dataframe, cols):
    '''
    Adds dummy variable columns to the dataframe for each unique value in the column
    and returns a new dataframe with dummy columns added and the input column dropped.
    '''
    return pd.get_dummies(dataframe, columns=cols)

