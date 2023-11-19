'''
functions needed to create NN models
'''
import os
import sys
from pathlib import Path
import pickle
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Lambda
from keras.regularizers import l1
from keras.optimizers.legacy import Adam
from brain_age_prediction.utils.chek_model_type import check_model_type

ROOT_PATH = Path(__file__).parent.parent.parent

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
    try:
        with open(os.path.join(ROOT_PATH,'brain_age_prediction','best_hyperparams',
                           'structural_model_hyperparams.pkl'), 'rb') as fp:
            s_best_hyperparams = pickle.load(fp)
    except OSError as e:
        print('Cannot load best hyperparams of the structural model:'
                f'cannot read the file in which they should be saved! \n{e}')
        sys.exit(1)
    try:
        with open(os.path.join(ROOT_PATH,'brain_age_prediction','best_hyperparams',
                            'functional_model_hyperparams.pkl'), 'rb') as fp:
            f_best_hyperparams = pickle.load(fp)
    except OSError as e:
        print('Cannot load best hyperparams of the functional model:'
                f'cannot read the file in which they should be saved! \n{e}')
        sys.exit(1)

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


def load_model(model_type):
    '''
    Load a saved keras model and compile it,
    return the compiled model
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    json_name = f'{model_type}_model.json'
    h5_name   = f'{model_type}_model_weights.h5'

    # load json and create model
    try:
        json_file         = open(os.path.join(ROOT_PATH,'brain_age_prediction',
                                    'saved_models',json_name), 'r', encoding='utf-8')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    except OSError as e:
        print('Cannot load the model:'
              f'cannot read the file in which the model should be saved! \n{e}')
        sys.exit(1)


    # load weights into new model
    try:
        model.load_weights(os.path.join(ROOT_PATH,'brain_age_prediction','saved_models',h5_name))
    except OSError as e:
        print('Cannot load weights into the model:'
              f'cannot read the file in which the weights should be saved! \n{e}')
        sys.exit(1)

    optim = Adam(learning_rate = 0.001)
    model.compile(loss='mae', optimizer=optim)
    print("Loaded model from disk")
    return model

