import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from keras.models import Sequential, Model
from keras.layers import Input, Dense,Dropout,BatchNormalization, concatenate
from keras.regularizers import l1
from keras.optimizers.legacy import Adam


def load_dataset(dataset_name):
    '''
    Load the dataset as pandas dataframes and return 2 different dataframes:
    - 1 for the TD group
    - 1 for the ASD group
    '''

    #import dataset
    file_path_structural = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'dataset-ABIDE-I-II',dataset_name))

    df = pd.read_csv(file_path_structural)
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

    for i in range(hidden_layers):
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
    - the number of hiddenlayers.

    it returns the compiled model using MAE as loss, and Adam with lr=0.001 as optimizer
    '''
    model = Sequential()
    model.add(Input(shape=(5253,)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    for i in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation='relu',kernel_regularizer=l1(0.01)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))

    #compile model
    optim = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optim)
    return model




def create_joint_model(hidden_neurons, hidden_layers):
    # create model
    model_f = Sequential()
    model_f.add(Dense(100, input_shape=(5253,), activation='relu',kernel_regularizer=l1(0.01)))
    model_f.add(Dropout(0.5))
    model_f.add(BatchNormalization())
    model_f.add(Dense(100, activation='relu',kernel_regularizer=l1(0.01)))
    model_f.add(Dropout(0.2))
    model_f.add(BatchNormalization())


    model_s = Sequential()
    model_s.add(Dense(50, input_shape=(221,), activation='relu',kernel_regularizer=l1(0.01)))
    model_s.add(Dropout(0.5))
    model_s.add(BatchNormalization())
    model_s.add(Dense(20, activation='relu',kernel_regularizer=l1(0.01)))
    model_s.add(Dropout(0.2))
    model_s.add(BatchNormalization())

    model_concat = concatenate([model_f.output, model_s.output], axis=-1)

    for i in range(hidden_layers):
        model_concat = Dense(hidden_neurons, activation='relu',kernel_regularizer=l1(0.01))(model_concat)
        model_concat = Dropout(0.2)(model_concat)
        model_concat = BatchNormalization()(model_concat)

    model_concat = Dense(1, activation='linear',kernel_regularizer=l1(0.01))(model_concat)

    model = Model(inputs=[model_f.input, model_s.input], outputs=model_concat)


    #compile the model
    optim = Adam(learning_rate=0.01)
    model.compile(loss='mae', optimizer=optim)

    return model




def line(x, a, b):
    '''
    model of a line
    '''
    return a*x +b
