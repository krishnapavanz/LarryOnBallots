import pandas as pd
import matplotlib.pylab as plt
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
        for n_estimators in [1, 5, 10, 20]:#, 40, 80, 160, 200, 250, 300, 400, 500]:
            for max_depth in [1, 5, 10, 20]:#, 40, 80, 160, 200, 250, 300, 400, 500]:
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


def predict_random_forest(parameters, X_train, y_train, X, random_state = True):
    """
    Make predictions with given parameters
    
    Inputs:
        - parameters (dict): {'criterion': best_criterion,
            'n_estimators': best_n_estimators, 'max_depth': best_max_depth}¨
        - X_train (pd.DataFrame): training features
        - y_train (pd.DataFrame): training labels
        - X (pd.DataFrame): Observation to predict (row)
        - random_state (bool): Whether we want to use a random state
    
    Output:
        - y (list of ints): predicted labels
    """
    if random_state:
        random_forest_classifier = RandomForestClassifier(max_depth = parameters['max_depth'],\
            n_estimators = parameters['n_estimators'], criterion = parameters['criterion'], random_state = 123)
    else:
        random_forest_classifier = RandomForestClassifier(max_depth = parameters['max_depth'],\
            n_estimators = parameters['n_estimators'], criterion = parameters['criterion'])
    
    random_forest_classifier.fit(X_train, y_train)

    return random_forest_classifier.predict(X)


def plot_random_forest(accuracies):
    """
    3D plot of the random forest's accuracies according to the max_depth,
    number of trees, and criterion.

    Input:
        - accuracies (pd.DataFrame): Accuracies of the random forest

    Output:
        - None (graph is displayed)
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
