'''
functions useful to load the datasets and preprocess data
'''
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split



def load_dataset(dataset_name):
    '''
    Load the dataset as pandas dataframes and return 2 different dataframes:
    - 1 for the TD group
    - 1 for the ASD group
    '''
    #import dataset
    try:
        path = Path(__file__)
        file_path = os.path.join(path.parent.parent.parent,'dataset-ABIDE-I-II',dataset_name)
        df = pd.read_csv(file_path)
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
