from useful_functions import load_dataset, preprocessing, create_model



if __name__ == "__main__":
    import numpy as np

    DATASET_NAME = 'Harmonized_structural_features.csv'
    df_s_td, df_s_asd = load_dataset(dataset_name=DATASET_NAME)

    #inputs
    X = preprocessing(df_s_td)
    #targets
    y = np.array(df_s_td['AGE_AT_SCAN'])

    from sklearn.model_selection import train_test_split

    # shuffle and split training and test sets
    SEED = 7 #for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=SEED)


    #perform model selection:
    from sklearn.model_selection import KFold
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from scikeras.wrappers import KerasRegressor
    from sklearn.model_selection import GridSearchCV

    k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)

    model = KerasRegressor(model=create_model, verbose=0)

    # define search space
    input_neurons = [10]
    hidden_neurons = [10]
    hidden_layers = [1,2,3]

    param_grid = {'model__input_neurons': input_neurons,
                   'model__hidden_neurons': hidden_neurons,
                    'model__hidden_layers': hidden_layers }

    search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    n_jobs=1,
                    scoring='neg_mean_absolute_error',
                    cv=k_fold,
                    verbose=0)

    grid_result = search.fit(X_train, y_train, epochs = 100, verbose =0)
    # get the best performing model fit on the whole training set
    best_model = grid_result.best_estimator_
    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f'{mean} ({stdev}) with: {param}')


    exit()
