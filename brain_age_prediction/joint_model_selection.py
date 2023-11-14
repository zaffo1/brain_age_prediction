from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import os
import pickle
from keras.models import Model, Sequential
from keras.layers import Dense,Dropout,BatchNormalization, concatenate, Input, Lambda
from keras.regularizers import l1
from keras.optimizers.legacy import Adam
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def load_model(functional=False,structural=False):
    '''
    Load the best model (functional/structural) obtained throug model selection.
    '''

    if structural:
        filename = 'structural_model_hyperparams.pkl'
        create_model = create_structural_model

    if functional:
        filename = 'functional_model_hyperparams.pkl'
        create_model = create_functional_model

    # Read dictionary pkl file
    with open(os.path.join('best_hyperparams',filename), 'rb') as fp:
        best_hyperparams = pickle.load(fp)


    model = create_model(dropout=best_hyperparams['model__dropout'],
                         hidden_neurons=best_hyperparams['model__hidden_neurons'],
                         hidden_layers=best_hyperparams['model__hidden_layers'])
    model.summary()
    return model

def join_models(hidden_units):
    '''
    join functional and structural using a concatenate layer. Add another hidden layer with a number of units
    equal to "hidden_units".
    Return the compiled joint model.
    '''



    combi_input = Input(shape=(5474,))
    f_input = Lambda(lambda x: (x[:,:5253]))(combi_input) # (None, 1)
    s_input = Lambda(lambda x: (x[:,5352:]))(combi_input)

    #model_f = Input(shape=(5253,))(f_input)
    model_f = (Dropout(0.5))(f_input)
    model_f = (BatchNormalization())(model_f)

    model_f = Dense(50, activation='relu',kernel_regularizer=l1(0.01))(model_f)
    # model.add(Dropout(dropout))
    #model.add(BatchNormalization())

    c = concatenate([model_f, s_input], axis=-1)
    #create joint model, removing the last layers of the single models

    #model_concat = concatenate([model_f.layers[-2].output, model_s.layers[-2].output], axis=-1)
    model_concat = Dense(hidden_units, activation='relu',kernel_regularizer=l1(0.01))(c)
    model_concat = Dropout(0.2)(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Dense(1, activation='linear',kernel_regularizer=l1(0.01))(model_concat)

    model = Model(inputs=combi_input, outputs=model_concat)

    #compile the model
    optim = Adam(learning_rate=0.01)
    model.compile(loss='mae', optimizer=optim)

    return model


def train(model,X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test):
    '''
    Train the joint model
    '''

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    MAX_EPOCHS = 1000

    train=model.fit([X_f_train,X_s_train],
                    y_train,
                    epochs=MAX_EPOCHS,
                    batch_size=32,
                    verbose=1,
                    validation_data=([X_f_test,X_s_test],y_test),
                    callbacks=[reduce_lr,early_stopping])

    #evalueate model
    score = model.evaluate([X_f_test,X_s_test], y_test, verbose=0)
    print(f'TEST MAE = {score}')

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models','joint_model.json'), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join('saved_models','joint_model_weights.h5'))
    print("Saved model to disk")

    # plot loss during training
    plt.plot(train.history['loss'], label='train')
    plt.plot(train.history['val_loss'], label='test')
    plt.title('Joint Model')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')
    plt.show()




def model_selection(search_space, X_train,y_train,n_folds=5):
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


    model = KerasRegressor(model=join_models, verbose=0)
    filename = 'joint_model_hyperparams.pkl'



    # define search space
    param_grid = {'model__hidden_units': search_space[0]}

    grid_search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    n_jobs=1,
                    scoring='neg_mean_absolute_error',
                    cv=k_fold,
                    verbose=3)


    max_epochs = 300
    grid_result = grid_search.fit(X_train, y_train, epochs = max_epochs, verbose = 0)

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
    from sklearn.model_selection import train_test_split


    SEED = 7

    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    X_s = preprocessing(df_s_td)
    X_f = preprocessing(df_f_td)

    #check if targets are equal
    y_s = df_s_td['AGE_AT_SCAN']
    y_f = df_f_td['AGE_AT_SCAN']
    print(y_s.equals(y_f))
    y = np.array(y_s)

    X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test = train_test_split(
        X_f,X_s, y, test_size=0.3, random_state=SEED)


    print(X_f_train.shape)
    print(X_s_train.shape)
    merged_inputs = np.concatenate([X_f_train,X_s_train], axis=-1)
    print(merged_inputs.shape)
    mf = merged_inputs[:,:5253]
    print(mf.shape)
    ms = merged_inputs[:,5253:]
    print(ms.shape)

    #join the two models and create a joint model:
    HIDDEN_UNITS = 5
    model = join_models(hidden_units=HIDDEN_UNITS)


    #plot the model architecture
    plot_model(model, "plots/architecture_joint_model.png", show_shapes=True)

    #k_fold_cross_validation(model,X_f_train, X_s_train, y_train)

    hidden_neurons = [10,20,30,50]
    search = [hidden_neurons]

    model_selection(search_space=search,X_train=merged_inputs,y_train=y_train)

    exit()
    #train the model
    train(model,X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test)
