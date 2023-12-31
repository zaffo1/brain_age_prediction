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

ROOT_PATH = Path(__file__).parent.parent.parent


def load_dataset(dataset_name):
    '''
    Load the dataset as pandas dataframes and return two different dataframes:
    - One for the TD group
    - One for the ASD group

    :param str dataset_name: The name of the dataset.

    :return: Two pandas dataframes for the TD and ASD groups, respectively.
    :rtype: tuple

    This function loads a dataset from a CSV file into a pandas dataframe. It then
    separates the dataframe into two based on the diagnostic group (TD or ASD).
    The resulting dataframes are returned as a tuple.
    '''
    #import dataset
    try:
        file_path = os.path.join(ROOT_PATH,'dataset-ABIDE-I-II',dataset_name)
        df = pd.read_csv(file_path)
    except OSError as e:
        print(f'Cannot load the dataset! \n{e}')
        sys.exit(1)

    df_td = df[(df["DX_GROUP"] == -1)]
    df_asd = df[(df["DX_GROUP"] == 1)]

    return df_td, df_asd


def preprocessing(df):
    '''
    Takes in input a pandas dataframe and returns a numpy array of the features used
    as input for the learning process. Additionally, it applies a RobustScaler preprocessing
    to the input features.

    :param pandas.DataFrame df: The input dataframe.

    :return: The preprocessed numpy array of features.
    :rtype: numpy.ndarray

    This function extracts relevant features from the input dataframe, converts them
    to a numpy array, and applies RobustScaler for preprocessing to handle outliers.
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
    Split the data into train and test according to the "split" variable.

    :param float split: The ratio of the dataset to include in the test split.
    :param int random_state: Seed for the random state of the train_test_split function
                             (for reproducibility).

    :return: Tuple containing training and test sets for structural and functional data.
    :rtype: tuple

    This function loads both structural and functional datasets, preprocesses the input
    features using the `preprocessing` function, and then splits the data into training
    and testing sets.
    '''

    df_s_td = load_dataset(dataset_name='Harmonized_structural_features.csv')[0]
    #load functional dataset
    df_f_td = load_dataset(dataset_name='Harmonized_functional_features.csv')[0]

    #preprocess input features
    x_s = preprocessing(df_s_td)
    x_f = preprocessing(df_f_td)

    #load targets (functional and structural are the same)
    y = np.array(df_s_td['AGE_AT_SCAN'])

    # shuffle and split training and test sets
    x_s_tr, x_s_te, y_s_tr, y_s_te = train_test_split(x_s, y, test_size=split,
                                                            random_state=seed)
    x_f_tr, x_f_te, y_f_tr, y_f_te = train_test_split(x_f, y, test_size=split,
                                                            random_state=seed)

    return x_s_tr, x_s_te, y_s_tr, y_s_te, x_f_tr, x_f_te, y_f_tr, y_f_te
