from random import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train, X_dev, y_train, y_dev, random_state = True):
    """
    Build several random forests by varying the following parameters:
        - n_estimators: number of trees
        - criterion: gini, entropy
        - max_depth: maximum depth of the tree
    
    Inputs:
        - X_train (pd.DataFrame): training features
        - X_dev (pd.DataFrame): development features
        - y_train (pd.DataFrame): training labels
        - y_dev (pd.DataFrame): development labels
        - random_state (bool): Whether we want to use a random state

    Outputs:
        - best_parameters (dict): {'criterion': best_criterion,
            'n_estimators': best_n_estimators, 'max_depth': best_max_depth}
        - best_accuracy (float): best development accuracy score
        - accuracies (pd.DataFrame): accuracies for every combination of hyperparameters
    """
    accuracies = pd.DataFrame(columns = ['criterion', 'n_estimators', 'max_depth', 'accuracy'])
    i = 0
    for criterion in ['gini', 'entropy']:
        for n_estimators in [1, 5, 10, 20, 40, 80, 160, 200, 250, 300, 400, 500]:
            for max_depth in [1, 5, 10, 20, 40, 80, 160, 200, 250, 300, 400, 500]:
                if random_state:
                    random_forest_classifier = RandomForestClassifier(max_depth = max_depth,\
                        n_estimators = n_estimators, criterion = criterion, random_state = 123)
                else:
                    random_forest_classifier = RandomForestClassifier(max_depth = max_depth,\
                        n_estimators = n_estimators, criterion = criterion)
                random_forest_classifier.fit(X_train, y_train)
                accuracy = random_forest_classifier.score(X_dev, y_dev)
                accuracies.loc[i] = [criterion, n_estimators, max_depth, accuracy]
                i += 1
    best_accuracy = max(accuracies['accuracy'])
    best_row = accuracies.loc[accuracies['accuracy'] == best_accuracy]
    best_parameters = {key: best_row.loc[int(best_row.index[0]), key] for key in ['criterion', 'n_estimators', 'max_depth']}
    return best_parameters, best_accuracy, accuracies 



"""
BELOW IS TO DELETE ONCE NURIA COPIES IT - See the last chunk of code
"""

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import binarize
import sys
import os
sys.path.append(os.path.abspath('../resources'))
import data_preprocessing as dp
import split_normalization as sn
file_dir = os.path.abspath('')
face_cov_filepath = os.path.join(file_dir,"..","data","face_covering.csv")
demographics_filepath = os.path.join(file_dir,"..","data","demographics_2021.csv")
demographics = dp.load_demographics(demographics_filepath)
face_covering = dp.load_referendum(face_cov_filepath)
merged_data = dp.merge_demographics_referendum(demographics, face_covering)
names = ['population_density', 'foreigner_percentage', \
    'age_percentage_between_20_64', 'agriculture_surface_perc', \
    'participation_rate', 'yes_perc']
df_filtered = merged_data.filter(names, axis=1)
df_filtered = df_filtered.dropna()
X = df_filtered.iloc[:, :-1].values
y = df_filtered.iloc[:, len(names)-1].values
y = np.array([1 if x > 50 else 0 for x in y])
X_train, X_test, X_dev, y_train, y_test, y_dev = sn.split(X, y)
accuracies = random_forest(X_train, X_dev, y_train, y_dev)[2]



"""
Below is the plot for the notebook
"""

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')

gini_accuracies = accuracies.loc[accuracies['criterion'] == 'gini']
ax.scatter(gini_accuracies['max_depth'], gini_accuracies['n_estimators'], gini_accuracies['accuracy'], alpha = 1, color = 'red', label = 'Gini')

entropy_accuracies = accuracies.loc[accuracies['criterion'] == 'entropy']
ax.scatter(entropy_accuracies['max_depth'], entropy_accuracies['n_estimators'], entropy_accuracies['accuracy'], alpha = 1, color = 'blue', label = 'Entropy')

ax.set_title("Accuracy According to Different Hyperparameters", fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 24})
ax.set_xlabel('Max Depth', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
ax.set_ylabel('Number of Trees', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
ax.set_zlabel('Accuracy', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
ax.legend()

ax.view_init(15, 60)
plt.show()