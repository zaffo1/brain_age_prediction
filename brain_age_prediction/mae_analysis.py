'''
Study the distribution of absolute errors to assert their significance
'''
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, ttest_ind, shapiro
from sklearn.model_selection import train_test_split
from brain_age_prediction.utils.loading_data import load_dataset, preprocessing
from brain_age_prediction.utils.line import line
from brain_age_prediction.utils.custom_models import load_model
from brain_age_prediction.utils.chek_model_type import check_model_type

ROOT_PATH = Path(__file__).parent.parent
SEED = 7

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the figure suptitle
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




def compute_mae(model,df_s,df_f,model_type):
    '''
    function that performs an analysis of the results obtained using the
    regression models obtained.
    In particular it computes MAE.
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    x_s = preprocessing(df_s)
    x_f = preprocessing(df_f)

    y = np.array(df_s['AGE_AT_SCAN'])

    _, x_f_test, _, x_s_test, _, y_test = train_test_split(
        x_f,x_s, y, test_size=0.3, random_state=SEED)

    if model_type == 'structural':
        x_test = x_s_test

    if model_type == 'functional':
        x_test = x_f_test

    if model_type == 'joint':
        x_test = [x_f_test,x_s_test]

    #evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f'TD TEST LOSS = {score}')

    y_pred = model.predict(x_test)
    print(y_pred.shape)
    print(y_test.shape)
    ae = np.abs(y_pred.ravel() - y_test.ravel())
    mae = np.mean(np.abs(y_pred.ravel() - y_test.ravel()))

    print(f'TD TEST MAE = {mae}')
    return ae


def plot_distributions(ae_1, ae_2, label1, label2):
    '''
    plot PAD distributions
    '''
    p_val = empirical_p_value(ae_1, ae_2)

    plt.figure('AE distributions', figsize=[9,7])

    bins = plt.hist(ae_1,bins=30,color='purple',
                     alpha=0.5, density=True, label= f'{label1.capitalize()} (mean = {np.mean(ae_1):.3})')[1]
    plt.hist(ae_2,bins=bins,color='green',
              alpha=0.5, density=True, label= f'{label2.capitalize()} (mean = {np.mean(ae_2):.3})')

    plt.xlabel('Absolute Error [years]')
    plt.ylabel('Relative Frequency')
    plt.legend()

    plt.title(f' Absolute Error Distributions\n'
              f' {label1.capitalize()} vs {label2.capitalize()} model (empirical p={p_val:.2})')
    plt.savefig(os.path.join(
        ROOT_PATH,'brain_age_prediction','plots',f'{label1}_vs_{label2}_distribution.pdf'))

    plt.show()



if __name__ == "__main__":
    import tensorflow as tf

    #check if GPU is available
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')


    #structural model
    print('--------STRUCTURAL MODEL---------')
    structural_model = load_model(model_type='structural')
    ae_s = compute_mae(structural_model,df_s_td,df_f_td,model_type='structural')

    #functional model
    print('--------FUNCTIONAL MODEL---------')
    functional_model = load_model(model_type='functional')
    ae_f = compute_mae(functional_model,df_s_td,df_f_td,model_type='functional')

    #joint model
    print('--------JOINT MODEL---------')
    joint_model = load_model(model_type='joint')
    ae_j = compute_mae(joint_model,df_s_td,df_f_td,model_type='joint')

    #compare distributions of absolute errors
    plot_distributions(ae_s,ae_j,'structural','joint')
    plot_distributions(ae_s,ae_f,'structural','functional')
    plot_distributions(ae_f,ae_j,'functional','joint')