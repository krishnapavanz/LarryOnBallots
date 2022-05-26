import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import binarize

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


import sys
import os

sys.path.append(os.path.abspath('../resources'))
import data_preprocessing as dp
import split_normalization as sn

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
    eval_dict = {"confusion_matrix":confusion_matrix(y_test, y_test_pred),\
        "classification_report":classification_report(y_test, y_test_pred),\
        "accuracy_score":accuracy_score(y_test, y_test_pred)}
    return eval_dict


def decision_tree_hp(X_train, X_dev, y_train, y_dev):
    """
    Function that takes the Train and Development data and finds the
    Decision Tree model with the best hyper parameters 'criterion' and
    'max_depth'

    Inputs: X_train, X_dev, y_train, y_dev

    Outputs: A tuple with a dictionary {"max_criterion":max_criterion,
        "max_depth_tree":max_depth_tree} and
        a Pandas dataframe acc_df that contains columns
        ['criterion', 'depth', 'accu_rate']

    """
    # Training Decision tree
    max_acc = 0
    max_criterion = ""
    max_depth_tree = None
    acc_df = pd.DataFrame(columns=['criterion', 'depth', 'accu_rate'])
    # Calculating accuracy for criterion {“gini”, “entropy”, “log_loss”}
    for crit in [“gini”, “entropy”, “log_loss”]:
        for depth in range(1, 7, 1):
            clf_model = DecisionTreeClassifier(criterion=crit, random_state=42, max_depth=depth)
            clf_model.fit(X_train,y_train)
            acc_rate = clf_model.score(X_dev, y_dev)
            acc_df = acc_df.append({'criterion':crit, 'depth': depth, 'accu_rate':acc_rate}, \
            ignore_index=True)
            if acc_rate > max_acc:
                max_acc = acc_rate
                max_criterion = crit
                max_depth_tree = depth
    return ({"max_criterion":max_criterion, \
        "max_depth_tree":max_depth_tree}, acc_df)

def decision_tree_max_hp(X_train, X_test, y_train, y_test, max_criterion, max_depth_tree):
    """
    Function that takes the Train and Test data and
     returns the evaluation parameters for Decision Tree analysis

    Inputs: X_train, X_test, y_train, y_test, max_criterion, max_depth_tree

    Outputs: A dictionary with the keys and values for
        "confusion_matrix", "classification_report" and "accuracy_score"
    """
    clf_model = DecisionTreeClassifier(criterion=crit, random_state=42, max_depth=depth)
    clf_model.fit(X_train,y_train)
    y_test_pred = clf_model.predict(X_test)
    eval_dict = {"confusion_matrix":confusion_matrix(y_test, y_test_pred),\
        "classification_report":classification_report(y_test, y_test_pred),\
        "accuracy_score":accuracy_score(y_test, y_test_pred)}
    return eval_dict


def logistic_reg_hp(X_train, y_train):
    """
    Function that takes the Train data and finds the
    Logistic Regression model with the best hyper parameters

    Inputs: X_train, y_train

    Outputs: A tuple with a dictionary {"max_params":grid_result.best_params_,
        "max_acc":grid_result.best_score_} and
        a Pandas dataframe acc_df that contains columns
        ['params', 'mean_acc']
    """
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    acc_df = pd.DataFrame(columns=['params', 'mean_acc'])
    params = grid_result.cv_results_['params']
    means = grid_result.cv_results_['mean_test_score']
    for param, mean in zip(params, means):
        acc_df = acc_df.append({'params':param, 'mean_acc': mean}, \
            ignore_index=True)
    return ({"max_params":grid_result.best_params_, \
        "max_acc":grid_result.best_score_}, acc_df)


def logistic_reg_max_hp(X_train, X_test, y_train, y_test, max_params):
    """
    Function that takes the Train and Test data and
     returns the evaluation parameters for Logistic regression analysis

    Inputs: X_train, X_test, y_train, y_test, max_params

    Outputs: A dictionary with the keys and values for
        "confusion_matrix", "classification_report" and "accuracy_score"
    """
    logreg = LogisticRegression(random_state=0, \
        C = max_params["C"], solver = max_params["solver"], penalty =max_params["penalty"])
    logreg.fit(X_train,y_train)
    y_test_pred = logreg.predict(X_test)
    eval_dict = {"confusion_matrix":confusion_matrix(y_test, y_test_pred),\
        "classification_report":classification_report(y_test, y_test_pred),\
        "accuracy_score":accuracy_score(y_test, y_test_pred)}
    return eval_dict
