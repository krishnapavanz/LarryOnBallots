import numpy as np
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

    ax.view_init(10, 20)
    plt.show()


def feature_importance_random_forest(X_train, y_train, parameters, random_state = True):
    """
    Feature importance based on mean decrease in impurity.
    """
    if random_state:
        random_forest_classifier = RandomForestClassifier(max_depth = parameters['max_depth'],\
            n_estimators = parameters['n_estimators'], criterion = parameters['criterion'], random_state = 123)
    else:
        random_forest_classifier = RandomForestClassifier(max_depth = parameters['max_depth'],\
            n_estimators = parameters['n_estimators'], criterion = parameters['criterion'])

    random_forest_classifier.fit(X_train, y_train)

    importances = random_forest_classifier.feature_importances_
    importances_std = np.std([forest.feature_importances_ for forest in random_forest_classifier.estimators_], axis = 0)

    forest_importances = pd.Series(importances, index = X_train.columns)
    forest_importances.sort_values(ascending = False, inplace = True)
    forest_importances.rename({
        'population': 'pop',
        'population_variation': 'pop_var',
        'population_density': 'pop_den',
        'foreigner_percentage': 'for_per',
        'age_percentage_between_20_64': 'age_bet_20_64',
        'age_percentage_more_64': 'age_over_64',
        'marriage_rate': 'marriage_rate',
        'divorce_rate': 'divorce_rate',
        'birth_rate': 'birth_rate',
        'private_households': 'private_hous',
        'avg_household_size': 'avg_hous_size',
        'total_surface': 'tot_surf',
        'housing_and_infrastructure_surface': 'hous_inf_surf',
        'housing_and_infrastructure_surface_variation': 'hous_inf_surf_var',
        'agriculture_surface_perc': 'agr_surf',
        'agriculture_variation_surface_perc': 'agr_surf_var',
        'forest_surface_perc': 'for_surf',
        'unproductive_surface_perc': 'unprod_surf',
        'employment_primary': ' emp_pri',
        'employment_secondary': 'emp_sec',
        'employment_tertiary': 'emp_ter',
        'empty_housing_units': 'empty_hous',
        'new_housing_units_per_capita': 'new_hous_pc',
        'PLR': 'party_PLR',
        'PDC': 'party_PDC',
        'PS': 'party_PS',
        'UDC': 'party_UDC',
        'PEV_PCS': 'party_PEV',
        'PVL': 'party_PVL',
        'PBD': 'party_PBD',
        'PST_Sol': 'party_PST',
        'PES': 'party_PES',
        'small_right_parties': 'party_sm_right',
        'canton_Aargau': 'canton_AG',
        'canton_Appenzell_Ausserrhoden': 'canton_AR',
        'canton_Appenzell_Innerrhoden': 'canton_AI',
        'canton_Basel_Landschaft': 'canton_BL',
        'canton_Basel_Stadt': 'canton_BS',
        'canton_Bern': 'canton_BE',
        'canton_Fribourg': 'canton_FR',
        'canton_Geneve': 'canton_GE',
        'canton_Glarus': 'canton_GL',
        'canton_Graubunden': 'canton_GR',
        'canton_Jura': 'canton_JU',
        'canton_Luzern': 'canton_LU',
        'canton_Neuchatel': 'canton_NE',
        'canton_Nidwalden': 'canton_NI',
        'canton_Obwalden': 'canton OW',
        'canton_Schaffhausen': 'canton_SH',
        'canton_Schwyz': 'canton_SZ',
        'canton_Solothurn': 'canton_SO',
        'canton_St_Gallen': 'canton_SG',
        'canton_Thurgau': 'canton_TG',
        'canton_Ticino': 'canton_TI',
        'canton_Uri': 'canton_UR',
        'canton_Wallis': 'canton_VS',
        'canton_Vaud': 'canton_VD',
        'canton_Zug': 'canton_ZG',
        'canton_Zurich': 'canton_ZH'
    }, inplace = True)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    forest_importances.plot.bar(yerr = importances_std, ax = ax)

    ax.set_title("Feature importance for the Random Forest using MDI", fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 24})
    ax.set_xlabel('Features', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
    ax.set_ylabel('Mean Decrease in Impurity', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})

    plt.rcParams.update({'font.size': 9})
    fig.tight_layout()

    plt.show()