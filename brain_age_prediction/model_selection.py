from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pickle


def model_selection(search_space, X_train,y_train,N_FOLDS=5,SEED=7,functional=False,structural=False):
    '''
    function that perform k-fold cross validation in order to do model selection,
    in particular, the parameters are:
    .........................
    '''
    k_fold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    if structural:
        model = KerasRegressor(model=create_structural_model, verbose=0)
        filename = 'structural_model_hyperparams.pkl'

    if functional:
        model = KerasRegressor(model=create_functional_model, verbose=0)
        filename = 'functional_model_hyperparams.pkl'

    # define search space
    input_neurons = search_space[0]
    hidden_neurons = search_space[1]
    hidden_layers = search_space[2]

    param_grid = {'model__input_neurons': input_neurons,
                    'model__hidden_neurons': hidden_neurons,
                    'model__hidden_layers': hidden_layers}

    search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    n_jobs=1,
                    scoring='neg_mean_absolute_error',
                    cv=k_fold,
                    verbose=3)

    grid_result = search.fit(X_train, y_train, epochs = 100, verbose = 0)

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

    if 0:
        #structural model
        print('--------STRUCTURAL MODEL---------')
        df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

        X = preprocessing(df_s_td)
        y = np.array(df_s_td['AGE_AT_SCAN'])

        from sklearn.model_selection import train_test_split

        # shuffle and split training and test sets
        SEED = 7 #for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)

        search = [[20,50,100],[20,50,100],[1,2,3]]
        model_selection(structural=True, search_space=search,X_train=X_train,y_train=y_train)


    if 1:
        #functional model
        print('--------FUNCTIONAL MODEL---------')

        df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

        X = preprocessing(df_f_td)
        y = np.array(df_f_td['AGE_AT_SCAN'])

        from sklearn.model_selection import train_test_split

        # shuffle and split training and test sets
        SEED = 7 #for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)

        search = [[100,200,500],[100,200,500],[1,2,3]]
        model_selection(functional=True, search_space=search,X_train=X_train,y_train=y_train)