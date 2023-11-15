import pickle
from useful_functions import load_train_test, create_functional_model, create_structural_model, create_joint_model
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model

SEED = 7 #fixed for reproducibility


def model_selection(search_space, X_train,y_train,n_folds=5,functional=False,structural=False, joint=False):
    '''
    function that performs k-fold cross validation in order to do model selection using grid search,
    in particular, the parameters are:

    search_space: a list of lists that defines the combination of possible hyperparameters
    X_train: input features
    y_train: targets
    n_folds: number of folds of the k-fold cross validarion (default=5)
    functional: if True, consider the functional model (default=False)
    structural: if True, consider the structural model (default=False)

    Finally, the function saves to file the optial hyperparameters found
    '''

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    if structural:
        model = KerasRegressor(model=create_structural_model, verbose=0)
        filename = 'structural_model_hyperparams.pkl'

    if functional:
        model = KerasRegressor(model=create_functional_model, verbose=0)
        filename = 'functional_model_hyperparams.pkl'

    if joint:
        model = KerasRegressor(model=create_joint_model, verbose=0)
        filename = 'joint_model_hyperparams.pkl'


    # define search space
    param_grid = {'model__dropout': search_space[0],
                    'model__hidden_neurons': search_space[1],
                    'model__hidden_layers': search_space[2],
                    'model__model_selection': [True]}

    grid_search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    n_jobs=1,
                    scoring='neg_mean_absolute_error',
                    cv=k_fold,
                    verbose=3)

    if structural:
        max_epochs = 100
    if functional or joint:
        max_epochs = 300
    grid_result = grid_search.fit(X_train, y_train, epochs = max_epochs, verbose = 0)

    if joint:
    #Only in the case of the joint model (which has a different input layer in the case
    # of model selection), create a plot showing the plot used for ms.
        joint_model = create_joint_model(dropout=grid_result.best_params_['model__dropout'],
                         hidden_neurons=grid_result.best_params_['model__hidden_neurons'],
                         hidden_layers=grid_result.best_params_['model__hidden_layers'],
                         model_selection=grid_result.best_params_['model__model_selection'])
        plot_model(joint_model, "plots/architecture_joint_model_selection.png", show_shapes=True)

    #print the results of the grid search:
    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print(f'{mean} ({stdev}) with: {param}')

    #save best hyperparams found
    with open(os.path.join('best_hyperparams',filename), 'wb') as fp:
        pickle.dump(grid_result.best_params_, fp)
        print('optimal hyperparameters saved successfully to file')



if __name__ == "__main__":

    import numpy as np
    import os

    #load data
    X_s_train, X_s_test, y_s_train, y_s_test,X_f_train, X_f_test, y_f_train, y_f_test  = load_train_test(split=0.3,seed=SEED)


    if 0:
        #structural model grid search
        print('--------STRUCTURAL MODEL--------')

        #define search space
        dropout = [0.2,0.5]
        hidden_neurons = [10,20,30,50]
        hidden_layers = [1,2,3,4,5]
        search = [dropout,hidden_neurons,hidden_layers]

        model_selection(structural=True, search_space=search,X_train=X_s_train,y_train=y_s_train)

    if 0:
        #functional model grid search
        print('--------FUNCTIONAL MODEL--------')

        #define search space
        dropout = [0.1,0.2,0.5]
        hidden_neurons = [50,100,200]
        hidden_layers = [1,2,3]
        search = [dropout,hidden_neurons,hidden_layers]

        model_selection(functional=True, search_space=search,X_train=X_f_train,y_train=y_f_train)

    if 1:
        #joint model grid search
        print('--------JOINT MODEL--------')

        ### check if targets are equal!!
        y_train = y_f_train
        merged_inputs = np.concatenate([X_f_train,X_s_train], axis=-1)


        dropout = [0.2,0.5]
        hidden_neurons = [10,20,30,50]
        hidden_layers = [1,2,3]
        search = [dropout, hidden_neurons, hidden_layers]

        model_selection(joint=True, search_space=search, X_train=merged_inputs, y_train=y_train)
