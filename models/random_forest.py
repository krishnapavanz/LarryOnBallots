from random import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train, X_dev, y_train, y_dev, random_state = True):
    """
    Build a random forest

    Parameters:
        - n_estimators: number of trees
        - criterion: gini, entropy, etc.
        - max_depth: maximum depth of the tree
        - random_state
    """
    accuracies = pd.DataFrame(columns = ['criterion', 'n_estimators', 'max_depth', 'accuracy'])
    i = 0
    for criterion in ['gini', 'entropy']:
        for n_estimators in [1, 5, 10, 20]:#, 40, 80, 160]:
            for max_depth in [1, 5, 10, 20]:#, 40, 80, 160]:
                if random_state:
                    random_forest_classifier = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, criterion = criterion, random_state = 123)
                else:
                    random_forest_classifier = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, criterion = criterion)
                random_forest_classifier.fit(X_train, y_train)
                accuracy = random_forest_classifier.score(X_dev, y_dev)
                accuracies.loc[i] = [criterion, n_estimators, max_depth, accuracy]
                i += 1
    best_accuracy = max(accuracies['accuracy'])
    best_row = accuracies.loc[accuracies['accuracy'] == best_accuracy]
    best_parameters = {key: best_row.loc[int(best_row.index[0]), key] for key in ['criterion', 'n_estimators', 'max_depth']}
    return best_parameters, best_accuracy, accuracies 

"""
BELOW IS TO DELETE ONCE NURIA COPIES IT
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
print(accuracies)



fig, ax = plt.subplots(figsize = (10, 8))
ax.scatter(accuracies['max_depth'], accuracies['accuracy'], color = accuracies['criterion'])        ### color not working
# ax.plot(x, y2, c = 'blue', label = 'y2') 
# ax.set_xlabel('x') 
# ax.set_ylabel('y')
# ax.set_title('Question 1')
#ax.legend()
plt.show()