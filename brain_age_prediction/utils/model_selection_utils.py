'''
Functions used to perform Model Selection
'''
import os
import sys
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
from brain_age_prediction.utils.custom_models import (create_functional_model,
                                                      create_structural_model,
                                                      create_joint_model)
from brain_age_prediction.utils.chek_model_type import check_model_type

ROOT_PATH = Path(__file__).parent.parent.parent
SEED = 7 #fixed for reproducibility

def print_grid_search_results(grid_result,filename):
    '''
    Prints the results of the grid search and saves them to a file.

    :param grid_result: The output of grid_search.fit.
    :param str filename: The name to assign to the saved file.

    This function prints the best score and parameters found during a grid search,
    along with the mean and standard deviation of test scores for each combination
    of hyperparameters. It also saves the best hyperparameters to a file.
    '''

    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print(f'{mean} ({stdev}) with: {param}')

    #save best hyperparams found
    try:
        with open(os.path.join(
            ROOT_PATH,'brain_age_prediction','best_hyperparams',filename), 'wb') as fp:
            pickle.dump(grid_result.best_params_, fp)
            print('optimal hyperparameters saved successfully to file')
    except OSError as e:
        print(f'Cannot save best hyperparameters! \n{e}')
        sys.exit(1)

def model_selection(search_space, x_train,y_train,model_type,max_epochs=300):
    '''
    Perform k-fold cross-validation for model selection using grid search.

    :param list search_space: A list of lists defining the combination of possible hyperparameters.
    :param numpy.ndarray x_train: Input features.
    :param numpy.ndarray y_train: Targets.
    :param str model_type: The type of model to perform model
                           selection for ('structural', 'functional', or 'joint').
    :param int max_epochs: Maximum number of training epochs (default=300).

    This function performs k-fold cross-validation using grid search to
    find the optimal hyperparameters
    for the specified model type. It saves the optimal hyperparameters to a file.
    '''

    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)

    filename = f'{model_type}_model_hyperparams.pkl'

    if model_type == 'structural':
        model = KerasRegressor(model=create_structural_model, verbose=0)
    if model_type == 'functional':
        model = KerasRegressor(model=create_functional_model, verbose=0)
    if model_type == 'joint':
        model = KerasRegressor(model=create_joint_model, verbose=0)

    # define search space
    if model_type in ('structural', 'functional'):
        param_grid = {'model__dropout': search_space[0],
                        'model__hidden_neurons': search_space[1],
                        'model__hidden_layers': search_space[2],}
    elif model_type == 'joint':
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

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=10, min_lr=0.00001)

    grid_result = grid_search.fit(x_train,
                                y_train,
                                epochs = max_epochs,
                                batch_size=64,
                                verbose = 0,
                                callbacks=[reduce_lr])

    if model_type == 'joint':
    #Only in the case of the joint model (which has a different input layer in the case
    # of model selection), create a plot showing the plot used for ms.
        joint_model = create_joint_model(dropout=grid_result.best_params_['model__dropout'],
                         hidden_neurons=grid_result.best_params_['model__hidden_neurons'],
                         hidden_layers=grid_result.best_params_['model__hidden_layers'],
                         model_selection=grid_result.best_params_['model__model_selection'])

        plot_model(joint_model, os.path.join(ROOT_PATH,
        'brain_age_prediction','plots','architecture_joint_model_selection.png'), show_shapes=True)

    print_grid_search_results(grid_result,filename)
