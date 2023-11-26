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
    Creates and compiles the structural model.

    :param float dropout: The dropout rate for regularization.
    :param int hidden_neurons: The number of neurons in each hidden layer.
    :param int hidden_layers: The number of hidden layers.

    :return: The compiled model with Mean Absolute Error (MAE) as the loss function
             and Adam optimizer with a learning rate of 0.001.
    :rtype: keras.models.Sequential

    This function constructs a sequential neural network model with dropout
    regularization, batch normalization, and specified hidden layers and neurons.
    The output layer has a linear activation function, and the model is compiled
    using the Mean Absolute Error (MAE) loss function and the Adam optimizer.
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
    Create and compile a functional model.

    :param float dropout: The dropout rate for regularization.
    :param int hidden_neurons: The number of neurons in each hidden layer.
    :param int hidden_layers: The number of hidden layers.

    :return: The compiled model with Mean Absolute Error (MAE) as the loss function
             and Adam optimizer with a learning rate of 0.001.
    :rtype: keras.models.Sequential

    This function constructs a sequential neural network model with dropout
    regularization, batch normalization, and specified hidden layers and neurons.
    The output layer has a linear activation function, and the model is compiled
    using the Mean Absolute Error (MAE) loss function and the Adam optimizer.
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

def load_best_hyperparams():
    '''
    Return the best hyperparameters found for both the structural and functional models.

    :return: Tuple containing the best hyperparameters for the structural and functional models.
    :rtype: tuple

    This function loads and returns the best hyperparameters found for both the structural and
    functional models.
    The hyperparameters are stored in separate pickle files.
    '''

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

    return s_best_hyperparams, f_best_hyperparams

def create_joint_model(dropout, hidden_neurons, hidden_layers, model_selection=False):
    '''
    Create and compile a joint model that combines structural and functional features.

    :param float dropout: The dropout rate for regularization.
    :param int hidden_neurons: The number of neurons in each hidden layer.
    :param int hidden_layers: The number of hidden layers.
    :param bool model_selection: If True, the model is created for model selection purposes.

    :return: The compiled joint model with Mean Absolute Error (MAE) as the loss function
             and Adam optimizer with a learning rate of 0.01.
    :rtype: keras.models.Model

    Create the joint model. It consists of two branches which are basically the structural and
    functional model, with hyperparameters equal to the ones individually
    selected during model selection.
    These two branches are joined using a concatenate layer.
    After the concatenate layer, add a number of hidden layers equal to 'hidden_layers', each
    one with a number of units equal to 'hidden_units'.
    A dropout equal to 'dropout' is also applied, and a batch normalisation.

    The input 'model_selection' assumes categorical values, and it indicates if the created model
    is to be used for model selection purposes.
    This needs to be specified because scikit learn wrappers
    employed to do model selection don't support multi input models.
    So in this case a workaround was needed:
    the firs layer is a single layer which takes the concatenated structural
    and functional features, then this layer is split through Lambda layers.
    At this point the structure is the same as described before.

    The model is compiled using the Mean Absolute Error (MAE)
    loss function and the Adam optimizer with a learning rate of 0.01.
    '''

    # load best hyperparams
    s_best_hyperparams, f_best_hyperparams = load_best_hyperparams()


    if model_selection:
        combi_input = Input(shape=(5474,))
        f_input = Lambda(lambda x: (x[:,:5253]))(combi_input)
        s_input = Lambda(lambda x: (x[:,5253:]))(combi_input)

        model_f = (Dropout(f_best_hyperparams['model__dropout']))(f_input)
        model_f = (BatchNormalization())(model_f)
    else:
        input_f= Input(shape=(5253,))
        model_f = (Dropout(f_best_hyperparams['model__dropout']))(input_f)
        model_f = (BatchNormalization())(model_f)

    for _ in range(f_best_hyperparams['model__hidden_layers']):
        model_f = Dense(f_best_hyperparams['model__hidden_neurons'],
                         activation='relu',kernel_regularizer=l1(0.01))(model_f)
        model_f = (Dropout(f_best_hyperparams['model__dropout']))(model_f)
        model_f = (BatchNormalization())(model_f)

    if model_selection:
        model_s = (Dropout(s_best_hyperparams['model__dropout']))(s_input)
        model_s = (BatchNormalization())(model_s)
    else:
        input_s= Input(shape=(221,))
        model_s = (Dropout(s_best_hyperparams['model__dropout']))(input_s)
        model_s = (BatchNormalization())(model_s)

    for _ in range(s_best_hyperparams['model__hidden_layers']):
        model_s = Dense(s_best_hyperparams['model__hidden_neurons'],
                         activation='relu',kernel_regularizer=l1(0.01))(model_s)
        model_s = (Dropout(s_best_hyperparams['model__dropout']))(model_s)
        model_s = (BatchNormalization())(model_s)

    model_concat = concatenate([model_f, model_s], axis=-1)

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
    model.compile(loss='mae', optimizer = Adam(learning_rate=0.01))

    return model


def load_model(model_type):
    '''
    Load a saved Keras model and compile it.

    :param str model_type: The type of model to load ('structural', 'functional', or 'joint').

    :return: The compiled Keras model.
    :rtype: keras.models.Model

    This function loads a saved Keras model and its weights based on the provided
    'model_type' (either 'structural', 'functional', or 'joint'). The model is then
    compiled using the Mean Absolute Error (MAE) loss function and the Adam optimizer
    with a learning rate of 0.001.

    Note: Make sure that the saved model files are present in the specified paths.
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
        with open(os.path.join(ROOT_PATH,'brain_age_prediction','saved_models',json_name),
                   'r', encoding='utf-8') as json_file:
            loaded_model_json = json_file.read()
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
