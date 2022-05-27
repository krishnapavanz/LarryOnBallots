from sklearn.neural_network import MLPClassifier


def fit_nn(X_train, X_dev, y_train, y_dev, random_state=True):
    """
    Build several neural networks by varying the following parameters:
        - activation function: ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        - number of layers
        - number of notes
    
    Inputs:
        - X_train (pd.DataFrame): training features
        - X_dev (pd.DataFrame): development features
        - y_train (pd.DataFrame): training labels
        - y_dev (pd.DataFrame): development labels
        - random_state (bool): Whether we want to use a random state

    Outputs:
        - best_parameters (dict): {'activation_function': best activation function,
            'n_layers': best_n_layers, 'n_nodes': best_n_nodes}
        - best_accuracy (float): best development accuracy score
        - accuracies (pd.DataFrame): accuracies for every combination of hyperparameters
    """
    accuracies = pd.DataFrame(columns = ['activation_function', 'n_layers', 'n_nodes', 'accuracy'])
    n_layers = [2,3,4,5,6,7,8,9,10]
    n_nodes = [2,3,4,5,6,7,8,9,10]
    
    i = 0
    for activation_function in ['logistic', 'tanh', 'relu']:
        for n_layer in n_layers:
            for n_node in n_nodes:
                if random_state:
                    classifier = MLPClassifier(solver='sgd', alpha=1e-4,
                        hidden_layer_sizes=(n_layer, n_node), activation = activation_function, random_state=123)
                else:
                    classifier = MLPClassifier(solver='sgd', alpha=1e-4,
                        hidden_layer_sizes=(n_layer, n_node), activation = activation_function)

                classifier.fit(X_train, y_train)
                accuracy = classifier.score(X_dev, y_dev)
                accuracies.loc[i] = [activation_function, n_layer, n_node, accuracy]
                i += 1
    
    best_accuracy = max(accuracies['accuracy'])
    best_row = accuracies.loc[accuracies['accuracy'] == best_accuracy]
    best_parameters = {key: best_row.loc[int(best_row.index[0]), key] for key in ['activation_function', 'n_layers', 'n_nodes']}

    return best_parameters, best_accuracy, accuracies 

best_model_nn = MLPClassifier(solver='sgd', alpha=1e-4,
                    hidden_layer_sizes=(best_parameters['n_layers'], best_parameters['n_nodes']), 
                    activation = best_parameters['activation_function'], random_state=123).fit(X_train, y_train)
            
predictions_nn = best_model_nn.predict(X_test)


def plot_nn(accuracies_nn):
    '''
    3D plot of the nn's accuracies according to the number of layers,
    number of nodes, and activation function.

    Input:
        - accuracies (pd.DataFrame): Accuracies of the random forest

    Output:
        - None (graph is displayed)
    '''
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')

    logistic_accuracies = accuracies.loc[accuracies['activation_function'] == 'logistic']
    ax.scatter(logistic_accuracies['n_layers'], logistic_accuracies['n_nodes'], logistic_accuracies['accuracy'], alpha = 1, color = 'black', label = 'Logistic')

    tanh_accuracies = accuracies.loc[accuracies['activation_function'] == 'tanh']
    ax.scatter(tanh_accuracies['n_layers'], tanh_accuracies['n_nodes'], tanh_accuracies['accuracy'], alpha = 1, color = 'blue', label = 'Tanh')

    relu_accuracies = accuracies.loc[accuracies['activation_function'] == 'relu']
    ax.scatter(relu_accuracies['n_layers'], relu_accuracies['n_nodes'], relu_accuracies['accuracy'], alpha = 1, color = 'red', label = 'Relu')


    ax.set_title("Accuracy According to Different Hyperparameters", fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 24})
    ax.set_xlabel('N Layers', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
    ax.set_ylabel('N Nodes', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
    ax.set_zlabel('Accuracy', fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 18})
    ax.legend()

    ax.view_init(15, 60)
    plt.show()