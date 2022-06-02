'''
Functions for K-Nearest Neighbors Models
'''

import matplotlib.pylab as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

import sys
import os

sys.path.append(os.path.abspath('../resources'))
#import data_preprocessing as dp
#import split_normalization as sn

# Assuming all data to be available

def knn_analysis_hp(X_train, X_dev, y_train, y_dev):
    """
    Function that takes the Train and Development data and finds the
    KNN model with the best hyper parameter 'k'

    Inputs: X_train, X_dev, y_train, y_dev

    Outputs: A tuple with a dictionary {"max_k":max_k} and
        a Pandas dataframe acc_df that contains columns
        ['k', 'accu_rate']

    """
    # Training KNN Classifier
    max_acc = 0
    max_k = 0
    acc_df = pd.DataFrame(columns=['k', 'accu_rate'])
    # Calculating accuracy for K values between 1 and 39
    for i in range(1, 40, 2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        acc_rate = knn.score(X_dev, y_dev)
        acc_df = acc_df.append({'k':i, 'accu_rate':acc_rate}, \
            ignore_index=True)
        if acc_rate > max_acc:
            max_acc = acc_rate
            max_k = i
    return ({"max_k":max_k}, acc_df)

def knn_analysis_max_k(X_train, X_test, y_train, y_test, max_k):
    """
    Function that takes the Train and Test data and
     returns the evaluation parameters for KNN analysis

    Inputs: X_train, X_test, y_train, y_test, max_k

    Outputs: A dictionary with the keys and values for
        "confusion_matrix", "classification_report" and "accuracy_score"
    """
    classifier = KNeighborsClassifier(n_neighbors=max_k)
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    '''
    eval_dict = {"confusion_matrix":confusion_matrix(y_test, y_test_pred),\
        "classification_report":classification_report(y_test, y_test_pred),\
        "accuracy_score":accuracy_score(y_test, y_test_pred)}
    '''
    return y_test_pred


def knn_k_plot(acc_df):
    """
    Function that returns a k vs. accuracy plot for
    a pre-defined KNN classifier model

    Inputs: A Pandas dataframe acc_df that contains columns
        ['k', 'accu_rate']

    Returns: The matplotlib.pylab object with the k vs. accuracy plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(acc_df["k"], acc_df["accu_rate"], color='red', \
            linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('K Value vs. Accuracy rate')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy rate')
    return plt