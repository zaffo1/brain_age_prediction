'''
Module containing useful function to perform the analysis
'''

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Lambda
from keras.regularizers import l1
from keras.optimizers.legacy import Adam


def load_dataset(dataset_name):
    '''
    Load the dataset as pandas dataframes and return 2 different dataframes:
    - 1 for the TD group
    - 1 for the ASD group
    '''

    #import dataset
    try:
        file_path_structural = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'dataset-ABIDE-I-II',dataset_name))
        df = pd.read_csv(file_path_structural)
    except OSError as e:
        print(f'Cannot load the dataset! \n{e}')
        sys.exit(1)

    df_td = df[(df["DX_GROUP"] == -1)]
    df_asd = df[(df["DX_GROUP"] == 1)]

    return df_td, df_asd


def preprocessing(df):
    '''
    takes in input a pandas dataframe
    it returns a numpy array of the features used as input for the learning process
    moreover, it applies a RobustScaler preprocessing to the input features
    '''
    features = df.drop(['FILE_ID','Database_Abide','SITE','AGE_AT_SCAN','DX_GROUP'],axis=1)
    features=features.apply(lambda x: x.astype(float))
    features = np.array(features)
    transformer = RobustScaler().fit(features)
    features = transformer.transform(features)

    return features


def load_train_test(split=0.3, seed=7):
    '''
    Load both the structural and functional datasets.
    Apply preprocessing to input features.
    Split the data in train and test, according to the "split" variable
    '''

    df_s_td = load_dataset(dataset_name='Harmonized_structural_features.csv')[0]
    #load functional dataset
    df_f_td = load_dataset(dataset_name='Harmonized_functional_features.csv')[0]

    #preprocess input features
    x_s = preprocessing(df_s_td)
    x_f = preprocessing(df_f_td)

    #load targets
    #check if targets are equal (UNIT TEST)
    y_s = df_s_td['AGE_AT_SCAN']
    y_f = df_f_td['AGE_AT_SCAN']
    print(y_s.equals(y_f))
    y = np.array(y_s)

    # shuffle and split training and test sets
    x_s_tr, x_s_te, y_s_tr, y_s_te = train_test_split(x_s, y, test_size=split,
                                                            random_state=seed)
    x_f_tr, x_f_te, y_f_tr, y_f_te = train_test_split(x_f, y, test_size=split,
                                                            random_state=seed)

    return x_s_tr, x_s_te, y_s_tr, y_s_te, x_f_tr, x_f_te, y_f_tr, y_f_te


def create_structural_model(dropout, hidden_neurons, hidden_layers):
    '''
    create (and compile) our model
    in order to do model selection, it takes in input 3 hyperparameters:
    - the number of input neurons,
    - the number of hidden neurons,
    - the number of hiddenlayers.

    it returns the compiled model using MAE as loss, and Adam with lr=0.001 as optimizer
    '''
    model = Sequential()
    model.add(Input(shape=(221,)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation='relu',kernel_regularizer=l1(0.01)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))

    #compile model
    optim = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optim)
    return model


def create_functional_model(dropout, hidden_neurons, hidden_layers):
    '''
    create (and compile) our model
    in order to do model selection, it takes in input 3 hyperparameters:
    - the number of input neurons,
    - the number of hidden neurons,
    - the number of hidden layers.

    it returns the compiled model using MAE as loss, and Adam with lr=0.001 as optimizer
    '''
    model = Sequential()
    model.add(Input(shape=(5253,)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation='relu',kernel_regularizer=l1(0.01)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))

    #compile model
    optim = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optim)
    return model

def create_joint_model(dropout, hidden_neurons, hidden_layers, model_selection=False):
    '''
    join functional and structural using a concatenate layer.
    Add another hidden layer with a number of units
    equal to "hidden_units".
    Return the compiled joint model.
    '''
    # Read dictionary pkl file
    with open(os.path.join('best_hyperparams','structural_model_hyperparams.pkl'), 'rb') as fp:
        s_best_hyperparams = pickle.load(fp)
    with open(os.path.join('best_hyperparams','functional_model_hyperparams.pkl'), 'rb') as fp:
        f_best_hyperparams = pickle.load(fp)


    f_dropout=f_best_hyperparams['model__dropout']
    f_hidden_neurons=f_best_hyperparams['model__hidden_neurons']
    f_hidden_layers=f_best_hyperparams['model__hidden_layers']

    s_dropout=s_best_hyperparams['model__dropout']
    s_hidden_neurons=s_best_hyperparams['model__hidden_neurons']
    s_hidden_layers=s_best_hyperparams['model__hidden_layers']

    if model_selection:
        combi_input = Input(shape=(5474,))
        f_input = Lambda(lambda x: (x[:,:5253]))(combi_input) # (None, 1)
        s_input = Lambda(lambda x: (x[:,5253:]))(combi_input)

        model_f = (Dropout(f_dropout))(f_input)
        model_f = (BatchNormalization())(model_f)
    else:
        input_f= Input(shape=(5253,))
        model_f = (Dropout(f_dropout))(input_f)
        model_f = (BatchNormalization())(model_f)

    for _ in range(f_hidden_layers):
        model_f = Dense(f_hidden_neurons, activation='relu',kernel_regularizer=l1(0.01))(model_f)
        model_f = (Dropout(f_dropout))(model_f)
        model_f = (BatchNormalization())(model_f)

    if model_selection:
        model_s = (Dropout(s_dropout))(s_input)
        model_s = (BatchNormalization())(model_s)
    else:
        input_s= Input(shape=(221,))
        model_s = (Dropout(f_dropout))(input_s)
        model_s = (BatchNormalization())(model_s)

    for _ in range(s_hidden_layers):
        model_s = Dense(s_hidden_neurons, activation='relu',kernel_regularizer=l1(0.01))(model_s)
        model_s = (Dropout(s_dropout))(model_s)
        model_s = (BatchNormalization())(model_s)


    model_concat = concatenate([model_f, model_s], axis=-1)
    #create joint model, removing the last layers of the single models

    #model_concat = concatenate([model_f.layers[-2].output, model_s.layers[-2].output], axis=-1)
    for _ in range(hidden_layers):
        model_concat = Dense(hidden_neurons, activation='relu',
                             kernel_regularizer=l1(0.01))(model_concat)
        model_concat = Dropout(dropout)(model_concat)
        model_concat = BatchNormalization()(model_concat)

    model_concat = Dense(1, activation='linear',kernel_regularizer=l1(0.01))(model_concat)

    if model_selection:
        model = Model(inputs=combi_input, outputs=model_concat)
    else:
        model = Model(inputs=[input_f,input_s], outputs=model_concat)


    #compile the model
    optim = Adam(learning_rate=0.01)
    model.compile(loss='mae', optimizer=optim)


    return model

def line(x, a, b):
    '''
    model of a line
    '''
    return a*x +b
