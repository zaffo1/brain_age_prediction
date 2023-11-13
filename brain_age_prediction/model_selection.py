import pickle
from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

SEED = 7 #fixed for reproducibility


def model_selection(search_space, X_train,y_train,n_folds=5,functional=False,structural=False):
    '''
    function that perform k-fold cross validation in order to do model selection,
    in particular, the parameters are:
    .........................
    '''
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    if structural:
        model = KerasRegressor(model=create_structural_model, verbose=0)
        filename = 'structural_model_hyperparams.pkl'

    if functional:
        model = KerasRegressor(model=create_functional_model, verbose=0)
        filename = 'functional_model_hyperparams.pkl'

    # define search space
    param_grid = {'model__dropout': search_space[0],
                    'model__hidden_neurons': search_space[1],
                    'model__hidden_layers': search_space[2]}

    grid_search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    n_jobs=1,
                    scoring='neg_mean_absolute_error',
                    cv=k_fold,
                    verbose=3)

    grid_result = grid_search.fit(X_train, y_train, epochs = 100, verbose = 0)

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
    from sklearn.model_selection import train_test_split

    #structural model
    print('--------STRUCTURAL MODEL---------')
    #load structutal dataset
    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

    #preprocess input features
    X = preprocessing(df_s_td)
    #load targets
    y = np.array(df_s_td['AGE_AT_SCAN'])

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)
    #define search space
    dropout = [0.2,0.5]
    hidden_neurons = [10,20,30,50]
    hidden_layers = [1,2,3,4,5]
    search = [dropout,hidden_neurons,hidden_layers]

    model_selection(structural=True, search_space=search,X_train=X_train,y_train=y_train)


    #functional model
    print('--------FUNCTIONAL MODEL---------')
    #load functional dataset
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    X = preprocessing(df_f_td)
    y = np.array(df_f_td['AGE_AT_SCAN'])

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)

    #define search space
    dropout = [0.2,0.5]
    hidden_neurons = [100,200,500]
    hidden_layers = [1,2,3]
    search = [dropout,hidden_neurons,hidden_layers]

    model_selection(functional=True, search_space=search,X_train=X_train,y_train=y_train)
