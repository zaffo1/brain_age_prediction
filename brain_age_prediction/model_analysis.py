from useful_functions import load_dataset, preprocessing, line
from tensorflow.keras.models import model_from_json
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, ttest_ind
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

SEED = 7


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
    for i in range(0,p):
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


def load_model(structural=False,functional=False, joint=False):
    '''
    Load a saved keras model, and compile it
    return the compiled model
    '''
    if structural:
        json_name = 'structural_model.json'
        h5_name   = 'structural_model_weights.h5'

    if functional:
        json_name = 'functional_model.json'
        h5_name   = 'functional_model_weights.h5'

    if joint:
        json_name = 'joint_model.json'
        h5_name   = 'joint_model_weights.h5'


    # load json and create model
    try:
        json_file         = open(os.path.join('saved_models',json_name), 'r', encoding='utf-8')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    except OSError as e:
        print(f'Cannot load the model:
               cannot read the file in which the model should be saved! \n{e}')
        sys.exit(1)


    # load weights into new model
    try:
        model.load_weights(os.path.join('saved_models',h5_name))
    except OSError as e:
        print(f'Cannot load weights into the model:
               cannot read the file in which the weights should be saved! \n{e}')
        sys.exit(1)

    optim = Adam(learning_rate = 0.001)
    model.compile(loss='mae', optimizer=optim)
    print("Loaded model from disk")
    return model


def model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd,structural=False,functional=False,joint=False):
    '''
    function that performs an analysis of the results obtained using the
    regression models obtained.
    In particular: ..........
    '''

    X_s = preprocessing(df_s_td)
    X_f = preprocessing(df_f_td)

    y = np.array(df_s_td['AGE_AT_SCAN'])

    X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test = train_test_split(
        X_f,X_s, y, test_size=0.3, random_state=SEED)

    if structural:
        X_test = X_s_test

    if functional:
        X_test = X_f_test

    if joint:
        X_test = [X_f_test,X_s_test]


    #evalueate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'TD TEST MAE = {score}')
    y_pred = model.predict(X_test)

    r_td, p_td = permutation_test(y_test,y_pred.ravel())
    print(f'r = {r_td} (p={p_td})')

    plt.figure(1, figsize=[12,5])
    if structural:
        plt.suptitle('Structural Model')
    if functional:
        plt.suptitle('Functional Model')
    if joint:
        plt.suptitle('Joint Model')

    plt.subplot(121)
    plt.title('TD')

    plt.scatter(y_test,y_pred, color='blue', alpha=0.7, label=f'r={r_td:.2}\nMAE = {score:.3} years')
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')

    #fit
    popt = curve_fit(line, y_test, y_pred.ravel())[0]
    a, b = popt
    print(f'a = {popt[0]}, b={popt[1]}')

    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.plot(x,line(x,a,b), color = 'red', linestyle='-', label=f'fit ($\\alpha$ = {a:.2}, $\\beta$ = {b:.2})')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')
    plt.legend()
    plt.subplot(122)

    plt.title('TD (Corrected)')
    y_correct = age_correction(a,b,cron=y_test,pred=y_pred.ravel())

    r_td_correct, p_td_correct =permutation_test(y_test,y_correct.ravel())
    print(f'r = {r_td_correct} (p={p_td_correct})')


    score_correct = model.evaluate(X_test, y_correct, verbose=0)

    #predicted age difference (corrected)
    pad_c_td = y_correct - y_test
    print(f'PAD_c for TD (test set) = {pad_c_td.mean()} (std {pad_c_td.std()})')

    plt.scatter(y_test,y_correct, color='green', alpha=0.7, label=f'r={r_td_correct:.2}\nMAE = {score_correct:.3} years\nPAD = {pad_c_td.mean():.2} years')
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age (corrected) [years]')

    plt.legend()
    if structural:
        plt.savefig('plots/td_structural_model.pdf')
    if functional:
        plt.savefig('plots/td_functional_model.pdf')
    if joint:
        plt.savefig('plots/td_joint_model.pdf')


    #ASD

    if structural:
        X_asd = preprocessing(df_s_asd)
    if functional:
        X_asd = preprocessing(df_f_asd)
    if joint:
        X_asd = [preprocessing(df_f_asd),preprocessing(df_s_asd)]


    y_asd = np.array(df_s_asd['AGE_AT_SCAN'])
    y_pred_asd = model.predict(X_asd)

    score_asd = model.evaluate(X_asd, y_asd, verbose=0)
    print(f'ASD MAE = {score_asd}')

    r_asd, p_asd =permutation_test(y_asd,y_pred_asd.ravel())
    print(f'r = {r_asd} (p={p_asd})')

    plt.figure(2, figsize=[12,5])
    if structural:
        plt.suptitle('Structural Model')
    if functional:
        plt.suptitle('Functional Model')
    if joint:
        plt.suptitle('Joint Model')

    plt.subplot(121)
    plt.title('ASD')
    #plt.scatter(y_test,y_pred, color='blue', alpha=0.7, label=f'TD, r={r_td:.2}')
    plt.scatter(y_asd,y_pred_asd, color='red', alpha =0.5, label=f'r={r_asd:.2}\nMAE = {score_asd:.3} years')
    plt.plot(x,x, color = 'grey', linestyle='--')


    y_correct_asd = age_correction(a,b,cron=y_asd,pred=y_pred_asd.ravel())
    r_asd_correct, p_asd_correct =permutation_test(y_asd,y_correct_asd.ravel())
    print(f'r = {r_asd_correct} (p={p_asd_correct})')

    score_correct_asd = model.evaluate(X_asd, y_correct_asd, verbose=0)
    pad_c_asd = ((y_pred_asd.ravel()-b)/a) - y_asd
    print(f'PAD_c for ASD = {pad_c_asd.mean()} (std = {pad_c_asd.std()})')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')
    plt.legend()

    plt.subplot(122)
    plt.title('ASD (Corrected)')

    plt.scatter(y_asd,y_correct_asd, color='purple', alpha=0.7, label=f'r={r_asd_correct:.2}\nMAE = {score_correct_asd:.3} years\nPAD = {pad_c_asd.mean():.2} years')
    plt.plot(x,x, color = 'grey', linestyle='--')

    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age (corrected) [years]')

    plt.legend()
    if structural:
        plt.savefig('plots/asd_structural_model.pdf')
    if functional:
        plt.savefig('plots/asd_functional_model.pdf')
    if joint:
        plt.savefig('plots/asd_joint_model.pdf')

    plt.figure(3)
    plt.scatter(y_asd, pad_c_asd)
    plt.hlines(y=0, xmin=0,xmax=40)

    t, p = ttest_ind(a=pad_c_asd, b=pad_c_td, equal_var=True)

    plt.figure(4)

    bins = plt.hist(pad_c_td,bins=30,color='blue', alpha=0.5, density=True, label='TD')[1]
    plt.hist(pad_c_asd,bins=bins,color='red', alpha=0.5, density=True, label='ASD')
    plt.xlabel('PAD [years]')
    plt.ylabel('Relative Frequency')
    plt.legend()
    print(np.var(pad_c_td),np.var(pad_c_asd))
    print(t, p)

    if structural:
        plt.title(f'Structural Model\nPAD distribution (t={t:.3}, p={p:.3})')
        plt.savefig('plots/PAD_distribution_structural_model.pdf')

    if functional:
        plt.title(f'Functional Model\nPAD distribution (t={t:.3}, p={p:.3})')
        plt.savefig('plots/PAD_distribution_functional_model.pdf')

    if joint:
        plt.title(f'Joint Model\nPAD distribution (t={t:.3}, p={p:.3})')
        plt.savefig('plots/PAD_distribution_joint_model.pdf')

    plt.show()




if __name__ == "__main__":
    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')


    #structural model
    print('--------STRUCTURAL MODEL---------')
    model             = load_model(structural=True)

    model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd,structural=True)


    #functional model
    print('--------FUNCTIONAL MODEL---------')
    model             = load_model(functional=True)

    model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd, functional=True)

    #joint model
    print('--------JOINT MODEL---------')
    model             = load_model(joint=True)

    model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd, joint=True)