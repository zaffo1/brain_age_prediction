'''
Given the ML models already trained,
Apply them to our analysis
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


def permutation_test(x,y,permutation_number=1000):
    '''
    returns correlation and p-value
    '''
    r = pearsonr(x,y)[0]
    #Copy one of the features:
    x_s = np.copy(x)
    #Initialize variables:
    permuted_r = []
    #Number of permutations:
    p=permutation_number
    #Initialize permutation loop:
    for _ in range(0,p):
    #Shuffle one of the features:
        np.random.shuffle(x_s)
        #Computed permuted correlations and store them in pR:
        permuted_r.append(pearsonr(x_s,y)[0])

    #Significance:
    p_val = len(np.where(np.abs(permuted_r)>=np.abs(r))[0])/p
    return r, p_val


def age_correction(a,b,cron, pred):
    '''
    returns the corrected age using the de Lange method
    '''
    return pred + (cron-(a*cron+b))



def td_analysis(model,df_s,df_f,model_type):
    '''
    function that performs an analysis of the results obtained using the
    regression models obtained.
    In particular: ..........
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    x_s = preprocessing(df_s)
    x_f = preprocessing(df_f)

    y = np.array(df_s['AGE_AT_SCAN'])

    x_f_train, x_f_test, x_s_train, x_s_test, y_train, y_test = train_test_split(
        x_f,x_s, y, test_size=0.3, random_state=SEED)

    if model_type == 'structural':
        x_test = x_s_test
        x_train = x_s_train

    if model_type == 'functional':
        x_test = x_f_test
        x_train = x_f_train

    if model_type == 'joint':
        x_test = [x_f_test,x_s_test]
        x_train = [x_f_train,x_s_train]

    #evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f'TD TEST MAE (no correction)= {score}')
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    r_td, p_td = permutation_test(y_test,y_pred.ravel())

    print(f'r = {r_td} (p={p_td})')

    plt.figure('TD age prediction', figsize=[14,6])

    plt.suptitle(f'{model_type.capitalize()} Model')

    plt.subplot(121)
    plt.title('TD (Train)')

    #plt.scatter(y_test,y_pred, color='blue', alpha=0.7,
    #             label=f'test set: r={r_td:.2}, MAE = {score:.3} years')

    plt.scatter(y_train,y_pred_train, color='blue', alpha=0.7, label='Train')

    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')

    #fit
    popt, _ = curve_fit(line, y_train, y_pred_train.ravel())
    a, b = popt
    print(f'a = {popt[0]}, b={popt[1]}')

    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.plot(x,line(x,a,b), color = 'red', linestyle='-',
              label=f'fit ($\\alpha$ = {a:.2}, $\\beta$ = {b:.2})')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')
    plt.legend()
    plt.subplot(122)

    plt.title('TD (Test) (Corrected)')
    y_correct = age_correction(a,b,cron=y_test,pred=y_pred.ravel())

    r_td_correct, p_td_correct =permutation_test(y_test,y_correct.ravel())
    print(f'r = {r_td_correct} (p={p_td_correct})')


    score_correct = model.evaluate(x_test, y_correct, verbose=0)

    #predicted age difference (corrected)
    pad_c_td = y_correct - y_test
    print(f'PAD_c for TD (test set) = {pad_c_td.mean()} (std {pad_c_td.std()})')

    plt.scatter(y_test,y_correct, color='green', alpha=0.7,
                 label=f'Test (corrected)\nr={r_td_correct:.2}\nMAE = {score_correct:.3} years'
                 f'\nPAD = {pad_c_td.mean():.2} years')
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age (corrected) [years]')

    plt.legend()
    plt.savefig(os.path.join(ROOT_PATH,
                'brain_age_prediction','plots',f'td_{model_type}_model.pdf'))

    return pad_c_td, popt

def asd_analysis(model,df_s,df_f,popt,model_type):
    '''
    Analysis of the ASD data
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

    score_asd = model.evaluate(x_asd, y_asd, verbose=0)
    print(f'ASD MAE = {score_asd}')

    r_asd, p_asd =permutation_test(y_asd,y_pred_asd.ravel())
    print(f'r = {r_asd} (p={p_asd})')

    plt.figure('ASD age prediction', figsize=[14,6])
    plt.suptitle(f'{model_type.capitalize()} Model')

    plt.subplot(121)
    plt.title('ASD')
    plt.scatter(y_asd,y_pred_asd, color='red', alpha =0.5,
                 label=f'r={r_asd:.2}\nMAE = {score_asd:.3} years')
    x = np.linspace(min(y_asd),max(y_asd),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')

    a, b = popt
    y_correct_asd = age_correction(a,b,cron=y_asd,pred=y_pred_asd.ravel())
    r_asd_correct, p_asd_correct =permutation_test(y_asd,y_correct_asd.ravel())
    print(f'r = {r_asd_correct} (p={p_asd_correct})')

    score_correct_asd = model.evaluate(x_asd, y_correct_asd, verbose=0)
    pad_c_asd = y_correct_asd - y_asd
    print(f'PAD_c for ASD = {pad_c_asd.mean()} (std = {pad_c_asd.std()})')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')
    plt.legend()

    plt.subplot(122)
    plt.title('ASD (Corrected)')

    plt.scatter(y_asd,y_correct_asd, color='purple', alpha=0.7,
                 label=f'r={r_asd_correct:.2}\nMAE = {score_correct_asd:.3} years'
                 f'\nPAD = {pad_c_asd.mean():.2} years')
    plt.plot(x,x, color = 'grey', linestyle='--')

    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age (corrected) [years]')

    plt.legend()
    plt.savefig(os.path.join(
        ROOT_PATH,'brain_age_prediction','plots',f'asd_{model_type}_model.pdf'))

    plt.figure('prova',figsize=[10,5])
    plt.scatter(y_asd,pad_c_asd, color='purple', marker='.')
    plt.axhline(y = 0, color = 'grey', linestyle = '--')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('PAD [years]')
    plt.title(f'{model_type}')
    plt.savefig(os.path.join(
        ROOT_PATH,'brain_age_prediction','plots',f'prova_PAD_{model_type}.pdf'))

    return pad_c_asd


def empirical_p_value(group1, group2, num_permutations=100000):
    '''
    empirical p value
    '''
    # Observed test statistic (difference in means)
    observed_statistic = np.mean(group2) - np.mean(group1)

    # Initialize array to store permuted test statistics
    permuted_statistics = np.zeros(num_permutations)

    # Perform permutations and calculate permuted test statistics
    for i in range(num_permutations):
        # Concatenate and randomly permute the data
        combined_data = np.concatenate((group1, group2))
        np.random.shuffle(combined_data)

        # Split permuted data into two groups
        permuted_group1 = combined_data[:len(group1)]
        permuted_group2 = combined_data[len(group1):]

        # Calculate test statistic for permuted data
        permuted_statistic = np.mean(permuted_group2) - np.mean(permuted_group1)

        # Store permuted statistic
        permuted_statistics[i] = permuted_statistic

    # Calculate p-value
    p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations

    print("Empirical p-value:", p_value)
    return p_value

def plot_distributions(pad_c_td, pad_c_asd, model_type):
    '''
    plot PAD distributions
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    # Shapiro-Wilk test
    #_, p_td = shapiro(pad_c_td)
    #_, p_asd = shapiro(pad_c_asd)
    #print(f'Shapiro test of normality: p value td: {p_td}, asd: {p_asd}')

    #check variances of the two distrivution
    print(f'variance td: {np.var(pad_c_td)}, variance asd: {np.var(pad_c_asd)}')

    #t, p = ttest_ind(a=pad_c_asd, b=pad_c_td, equal_var=True)

    # empirical p value
    p_val = empirical_p_value(pad_c_td, pad_c_asd)

    plt.figure('PAD distributions', figsize=[9,7])

    bins = plt.hist(pad_c_td,bins=30,color='blue',
                     alpha=0.5, density=True, label= f'TD (mean = {np.mean(pad_c_td):.2})')[1]
    plt.hist(pad_c_asd,bins=bins,color='red',
              alpha=0.5, density=True, label= f'ASD (mean = {np.mean(pad_c_asd):.2})')
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


    #structural model
    print('--------STRUCTURAL MODEL---------')
    structural_model = load_model(model_type='structural')

    pad_td, fit_results = td_analysis(structural_model,df_s_td,df_f_td,model_type='structural')
    pad_asd = asd_analysis(structural_model,df_s_asd,df_f_asd,fit_results,model_type='structural')
    plot_distributions(pad_c_asd=pad_asd, pad_c_td=pad_td,model_type='structural')

    #functional model
    print('--------FUNCTIONAL MODEL---------')
    functional_model = load_model(model_type='functional')

    pad_td, fit_results = td_analysis(functional_model,df_s_td,df_f_td,model_type='functional')
    pad_asd = asd_analysis(functional_model,df_s_asd,df_f_asd,fit_results,model_type='functional')
    plot_distributions(pad_c_asd=pad_asd, pad_c_td=pad_td,model_type='functional')

    #joint model
    print('--------JOINT MODEL---------')
    joint_model = load_model(model_type='joint')

    pad_td, fit_results = td_analysis(joint_model,df_s_td,df_f_td,model_type='joint')
    pad_asd = asd_analysis(joint_model,df_s_asd,df_f_asd,fit_results,model_type='joint')
    plot_distributions(pad_c_asd=pad_asd, pad_c_td=pad_td,model_type='joint')
