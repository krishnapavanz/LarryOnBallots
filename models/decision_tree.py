import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import binarize
from sklearn import tree
from sklearn.tree import export_text
import graphviz
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

import sys
import os

sys.path.append(os.path.abspath('../resources'))
#import data_preprocessing as dp
#import split_normalization as sn

# Assuming all data to be available

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
    criteria = ["gini", "entropy"]
    for crit in criteria:
        for depth in range(1, len(X_train.columns)+1, 1):
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
    clf_model = DecisionTreeClassifier(criterion=max_criterion, random_state=42, max_depth=max_depth_tree)
    clf_model.fit(X_train,y_train)
    y_test_pred = clf_model.predict(X_test)
    '''
    eval_dict = {"confusion_matrix":confusion_matrix(y_test, y_test_pred),\
        "classification_report":classification_report(y_test, y_test_pred),\
        "accuracy_score":accuracy_score(y_test, y_test_pred)}
    '''
    return (y_test_pred, clf_model)

def decision_tree_depth_plot(acc_df):
    """
    Function that returns a depth vs. accuracy plot for
    a pre-defined decision tree classifier model

    Inputs: 
        - A Pandas dataframe acc_df that contains columns
          ['criterion', 'depth', 'accu_rate']

    Returns: The matplotlib.pylab object with the depth vs. accuracy plot
    """
    acc_df_gini = acc_df[acc_df.criterion == 'gini']
    acc_df_entropy = acc_df[acc_df.criterion == 'entropy']
   
    plt.figure(figsize=(12, 6))
    plt.plot(acc_df_gini["depth"], acc_df_gini["accu_rate"], color='red', \
            linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=6, label = 'gini')

    plt.plot(acc_df_entropy["depth"], acc_df_entropy["accu_rate"], color='green', \
            linestyle='dashed', marker='o',
            markerfacecolor='yellow', markersize=6, label = 'entropy')
    plt.title('Depth vs. Accuracy rate')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.show()
    return plt

def plot_best_decision_tree(clf_object, X_columns):
    '''
    Function that returns a graphical and textual depiction of the best
    decision tree.  
    Inputs:

    Returns:
    '''
    print("Textual model: \n")
    r = export_text(clf_object, feature_names = X_columns)
    print (r)

    # Plotting the decision tree
    # target = ['No', 'Yes']
    # dot_data = tree.export_graphviz(clf_object,
    #                                out_file = None,
    #                                feature_names = X_columns,
    #                                filled = True, 
    #                                rounded = True,
    #                                special_characters = True)
    # graph = graphviz.Source(dot_data)
    # print(graph)



