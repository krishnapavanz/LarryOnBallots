import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train, X_dev, y_train, y_dev):
    """
    Build a random forest

    Parameters:
        - n_estimators: number of trees
        - criterion: gini, entropy, etc.
        - max_depth: maximum depth of the tree
        - random_state
    """
    # best_accuracy = 0           ### TAKE THIS OUT AND JUST WORK WITH DATAFRAME
    # best_parameters = {}        ### SAME
    accuracies = pd.DataFrame(columns = ['criterion', 'n_estimators', 'max_depth', 'accuracy'])
    i = 0
    for criterion in ['gini', 'entropy']:
        for n_estimators in [1, 5, 10, 20]:#, 40, 80, 160]:
            for max_depth in [1, 5, 10, 20]:#, 40, 80, 160]:
                random_forest_classifier = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, criterion = criterion)
                random_forest_classifier.fit(X_train, y_train)
                accuracy = random_forest_classifier.score(X_dev, y_dev)
                accuracies.loc[i] = [criterion, n_estimators, max_depth, accuracy]
                # i += 1                                              ### SAME
                # if accuracy > best_accuracy:                        ### SAME
                #     best_accuracy = accuracy                        ### SAME
                #     best_parameters['criterion'] = criterion        ### SAME
                #     best_parameters['n_estimators'] = n_estimators  ### SAME
                #     best_parameters['max_depth'] = max_depth        ### SAME
                # print('\n', 'criterion: ', criterion, '\tn_estimators: ', n_estimators, '\tmax_depth: ', max_depth)
                # print(accuracy)
    best_accuracy = max(accuracies['accuracy'])
    print('here', best_accuracy)
    best_row = accuracies.loc[accuracies['accuracy'] == max(accuracies['accuracy'])]
    best_parameters = {key, best_row[key] for key in ['criterion', 'n_estimators', 'max_depth']}
    # best_accuracy = best_row['accuracy']
    best_parameters = 0
    return best_parameters, best_accuracy, accuracies 
