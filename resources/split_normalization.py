'''
Split and normalize the data
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def split(X, y, test_size = 0.1, dev_size = 0.1, random_state = 12, shuffle = True):
    '''
    Split the data into train, test, and developpment sets

    Inputs:
        - X (pd.DataFrame): feature data
        - y (pd.DataFrame): target data
        - test_size (float): test set size (default: 0.1)
        - dev_size (float): developpment set size (default: 0.1)
        - random_state (int): seed (default: 12)
        - shuffle (bool): shuffle the data before splitting (default: True)
    
    Outputs;
        - X_train (pd.DataFrame): train features
        - X_test (pd.DataFrame): test features
        - X_dev (pd.DataFrame): developpment features
        - y_train (pd.DataFrame): train outcome
        - y_test (pd.DataFrame): test outcome
        - y_dev (pd.DataFrame): developpment outcome
    '''
    X_train, X_test_dev, y_train, y_test_dev = \
        train_test_split(X, y, random_state = random_state, test_size = (test_size + dev_size))
    X_test, X_dev, y_test, y_dev = \
        train_test_split(X_test_dev, y_test_dev, random_state = random_state, test_size = (dev_size / (test_size + dev_size)))
    return X_train, X_test, X_dev, y_train, y_test, y_dev


def min_max_scaling(X_train, X_test):
    '''
    Apply min-max scaling to the feature data

    Inputs:
        - X_train (pd.DataFrame): train features
        - X_test (pd.DataFrame): test features
    
    Outputs:
        - X_train_reg (pd.DataFrame): normalized train features
        - X_test_reg (pd.DataFrame): normalized test features
    '''
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_reg = pd.DataFrame(scaler.transform(X_train))
    X_test_reg = pd.DataFrame(scaler.transform(X_test))
    return X_train_reg, X_test_reg


def standard_scaling(X_train, X_test):
    '''
    Apply standard scaling to the feature data

    Inputs:
        - X_train (pd.DataFrame): train features
        - X_test (pd.DataFrame): test features
    
    Outputs:
        - X_train_reg (pd.DataFrame): normalized train features
        - X_test_reg (pd.DataFrame): normalized test features
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_reg = pd.DataFrame(scaler.transform(X_train))
    X_test_reg = pd.DataFrame(scaler.transform(X_test))
    return X_train_reg, X_test_reg
