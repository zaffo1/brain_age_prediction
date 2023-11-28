'''
Given the ML models already trained,
Apply them to our analysis
'''
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from brain_age_prediction.utils.loading_data import load_dataset, preprocessing
from brain_age_prediction.utils.custom_models import load_model
from brain_age_prediction.utils.chek_model_type import check_model_type
from brain_age_prediction.utils.stats_utils import empirical_p_value, correlation

ROOT_PATH = Path(__file__).parent.parent
SEED = 7

SMALL_SIZE = 16
MEDIUM_SIZE = 17
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the figure suptitle
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def td_analysis(model,df_s,df_f,model_type):
    '''
    Analyze TD data using the regression model.

    :param keras.models.Sequential model: The trained Keras regression model.
    :param pd.DataFrame df_s: Dataframe containing structural features for TD group.
    :param pd.DataFrame df_f: Dataframe containing functional features for TD group.
    :param str model_type: Type of the model ('structural', 'functional', or 'joint').
    :return: Predicted Age Difference (PAD) for TD group.
    :rtype: numpy.ndarray
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

    y_pred = model.predict(x_test)
    pad_td = y_pred.ravel() - y_test
    mae_td = np.mean(np.abs(y_pred.ravel() - y_test.ravel()))
    r_td, p_td = correlation(y_test,y_pred.ravel())

    print(f'r = {r_td} (p={p_td})')

    plt.figure('TD age prediction', figsize=[7,6])
    plt.title(f'TD (test) {model_type.capitalize()} Model')

    plt.scatter(y_test,y_pred, color='blue', marker='.',
                 label=f'r={r_td:.2} (p < 0.001)\nMAE = {mae_td:.2} years'
                 f'\nPAD = {pad_td.mean():.2} years')

    x = np.linspace(min(y_test), max(y_test), 100)
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')

    plt.legend()
    plt.savefig(os.path.join(ROOT_PATH,
                'brain_age_prediction','plots',f'td_{model_type}_model.pdf'))

    return pad_td

def asd_analysis(model,df_s,df_f,model_type):
    '''
    Analyze ASD data using the regression model.

    :param keras.models.Sequential model: The trained Keras regression model.
    :param pd.DataFrame df_s: Dataframe containing structural features for ASD group.
    :param pd.DataFrame df_f: Dataframe containing functional features for ASD group.
    :param str model_type: Type of the model ('structural', 'functional', or 'joint').
    :return: Predicted Age Difference (PAD) for ASD group.
    :rtype: numpy.ndarray
    '''

    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    #ASD

    if model_type == 'structural':
        x_asd = preprocessing(df_s)
    if model_type == 'functional':
        x_asd = preprocessing(df_f)
    if model_type == 'joint':
        x_asd = [preprocessing(df_f),preprocessing(df_s)]


    y_asd = np.array(df_s['AGE_AT_SCAN'])
    y_pred_asd = model.predict(x_asd)
    pad_asd = y_pred_asd.ravel() - y_asd
    mae_asd = np.mean(np.abs(y_pred_asd.ravel() - y_asd.ravel()))
    r_asd, p_asd = correlation(y_asd,y_pred_asd.ravel())

    print(f'r = {r_asd} (p={p_asd})')

    plt.figure('ASD age prediction', figsize=[7,6])
    plt.title(f'ASD {model_type.capitalize()} Model')

    plt.scatter(y_asd,y_pred_asd, color='red', marker='.',
                 label=f'r={r_asd:.2} (p < 0.001)\nMAE = {mae_asd:.3} years'
                 f'\nPAD = {pad_asd.mean():.2} years')
    x = np.linspace(min(y_asd), max(y_asd), 100)
    plt.plot(x,x, color = 'grey', linestyle='--')

    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')

    plt.legend()
    plt.savefig(os.path.join(
        ROOT_PATH,'brain_age_prediction','plots',f'asd_{model_type}_model.pdf'))

    return pad_asd


def plot_distributions(pad_td, pad_asd, model_type):
    '''
    Plot Predicted Age Difference (PAD) distributions.

    :param numpy.ndarray pad_td: PAD for TD group.
    :param numpy.ndarray pad_asd: PAD for ASD group.
    :param str model_type: Type of the model ('structural', 'functional', or 'joint').
    '''

    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    print(f'variance td: {np.var(pad_td)}, variance asd: {np.var(pad_asd)}')

    # empirical p value
    p_val = empirical_p_value(pad_td, pad_asd)

    plt.figure('PAD distributions', figsize=[9,7])

    bins = plt.hist(pad_td, bins=30, color='blue', alpha=0.5, density=True,
                     label= f'TD (test) (mean = {np.mean(pad_td):.2} years)')[1]
    plt.hist(pad_asd,bins=bins,color='red',
              alpha=0.5, density=True, label= f'ASD (mean = {np.mean(pad_asd):.2} years)')
    plt.xlabel('PAD [years]')
    plt.ylabel('Relative Frequency')
    plt.legend()

    plt.title(f'{model_type.capitalize()} Model\nPAD distribution (empirical p={p_val:.2})')
    plt.savefig(os.path.join(
        ROOT_PATH,'brain_age_prediction','plots',f'PAD_distribution_{model_type}_model.pdf'))

    plt.show()



if __name__ == "__main__":
    import tensorflow as tf

    #check if GPU is available
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    for model_name in ('structural','functional','joint'):
        print(model_name.capitalize())
        mod = load_model(model_type=model_name)
        td_pad = td_analysis(mod,df_s_td,df_f_td,model_type=model_name)
        asd_pad = asd_analysis(mod,df_s_asd,df_f_asd,model_type=model_name)
        plot_distributions(pad_asd=asd_pad, pad_td=td_pad,model_type=model_name)
