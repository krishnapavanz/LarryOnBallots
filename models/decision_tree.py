import numpy as np
import matplotlib.pylab as plt
import pandas as pd
#import pydotplus # I added this to display the decision tree grapyh (Bill)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
import os

sys.path.append(os.path.abspath('../resources'))
import resources.data_preprocessing as dp  #I added 'resources.' to file path (Bill)
import resources.split_normalization as sn #I added 'resources.' to file path (Bill)

file_dir = os.path.dirname(__file__)

face_cov_filepath = os.path.join(file_dir,"..","data","face_covering.csv")
demographics_filepath = os.path.join(file_dir,"..","data","demographics_2021.csv")

# Loading data
demographics = dp.load_demographics(demographics_filepath)
face_covering = dp.load_referendum(face_cov_filepath)

# Merging data
merged_data = dp.merge_demographics_referendum(demographics, face_covering)

# 'names' can be edited to select a different set of columns
names = ['population_density', 'foreigner_percentage', \
    'age_percentage_between_20_64', 'agriculture_surface_perc', \
    'participation_rate', 'yes_perc']
df_filtered = merged_data.filter(names, axis=1)
df_filtered = df_filtered.dropna()

# Filtering data
X = df_filtered.iloc[:, :-1].values
y = df_filtered.iloc[:, len(names)-1].values

# Converting continuous values to binary
# Assuming that a referendum is passed when yes_perc >= 51
y = np.array([1 if x >=51 else 0 for x in y])


# Splitting data into train, development and test
X_train, X_test, X_dev, y_train, y_test, y_dev = sn.split(X, y)


# Training Decision tree
clf_model = DecisionTreeClassifier(criterion="gini", 
                                   random_state=42, 
                                   max_depth=3, 
                                   min_samples_leaf=5)
clf_model.fit(X_train,y_train)


## Development data
print("\nDevelopment data: \n")
# Testing with development data
y_dev_pred = clf_model.predict(X_dev)
print("Confusion matrix: \n", confusion_matrix(y_dev, y_dev_pred))
print("Classification report: \n",classification_report(y_dev, y_dev_pred))
print("Accuracy score: \n",accuracy_score(y_dev, y_dev_pred))

# Plotting the Decision tree
#target = [0, 1]
target = ['NO', 'YES'] # Updated to be string vice int type. (Bill)
feature_names = names[:-1]

# Graphical model
dot_data = tree.export_graphviz(clf_model,
                                out_file=None,
                                #out_file='tree_graph_test.png',
                                feature_names=feature_names,
                                class_names=target,
                                filled=True, rounded=True,      
                                special_characters=True)
graph = graphviz.Source(dot_data)
print('type_graph:', type(graph))
print(graph)
#graph.render('dtree_render_', view=True)
#graph.write_jpg('tree_graph_test')
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Textual model
r = export_text(clf_model, feature_names=feature_names)
print(r)


## Test data
# Testing with test data
print("\nTest data: \n")
y_test_pred = clf_model.predict(X_test)
print("Confusion matrix: \n", confusion_matrix(y_test, y_test_pred))
print("Classification report: \n",classification_report(y_test, y_test_pred))
print("Accuracy score: \n",accuracy_score(y_test, y_test_pred))

# Plotting the Decision tree
target = ["0", "1"]
feature_names = names[:-1]

# Graphical model
dot_data = tree.export_graphviz(clf_model,
                                out_file=None,
                      feature_names=feature_names,
                      class_names=target,
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
print(graph)

# Textual model
print("Textual model: \n")
r = export_text(clf_model, feature_names=feature_names)
print(r)
