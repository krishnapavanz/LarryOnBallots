'''
Functions for the logistic regression model.
'''

import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


def logistic_reg_hp(X_train, X_dev, y_train, y_dev):
    """
    Function that takes the Train data and finds the
    Logistic Regression model with the best hyper parameters

    Inputs: X_train, y_train

    Outputs: A tuple with a dictionary {"max_params":grid_result.best_params_,
        "max_acc":grid_result.best_score_} and
        a Pandas dataframe acc_df that contains columns
        ['params', 'mean_acc']
    """
    # Append train and dev data for this model as
    # RepeatedStratifiedKFold uses Cross-validation to
    # come up with best hyper-parameters for the model

    X_train = X_train.append(X_dev, ignore_index=True)
    y_train = y_train.append(y_dev, ignore_index=True)

    model = LogisticRegression(max_iter=10000)
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
    '''
    eval_dict = {"confusion_matrix": confusion_matrix(y_test, y_test_pred), \
        "classification_report": classification_report(y_test, y_test_pred), \
        "accuracy_score": accuracy_score(y_test, y_test_pred)}
    '''
    return (y_test_pred, logreg)


def logistic_reg_max_hp_coef(X_train, y_train, max_params):
    """
    Function that takes the Train and Test data and
     returns the coefficients and the intercept for
        the Logistic regression model

    Inputs: X_train, X_test, y_train, y_test, max_params

    Returns: A tuple with coefficients and the intercept for
        the model
    """
    logreg = LogisticRegression(random_state=0, \
        C = max_params["C"], solver = max_params["solver"], \
        penalty =max_params["penalty"])
    logreg.fit(X_train,y_train)
    return (logreg.coef_, logreg.intercept_)


def logistic_reg_plot(attr, logreg):
    '''
    Plot attributes and corresponding weights of given model. 
    '''

    df = pd.DataFrame(
        {'attribute': attr,
        'weight': [round(num, 1) for num in logreg.coef_[0]]
        })
    df = df.sort_values(by=['weight'], ascending=False)

    fig = px.bar(df, y='weight', x='attribute', text_auto='.2s',
                title="Weights of Logistic Regression",
                width=1000, height=800)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    
    return fig

