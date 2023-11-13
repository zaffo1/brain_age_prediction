from useful_functions import load_dataset, preprocessing, line
from tensorflow.keras.models import model_from_json
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import numpy as np
import os

SEED = 7

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
    json_file         = open(os.path.join('saved_models',json_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join('saved_models',h5_name))

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
    r_td, _ = pearsonr(y_test,y_pred.ravel())

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
    popt, pcov = curve_fit(line, y_test, y_pred.ravel())
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
    r_td_correct, _ = pearsonr(y_test,y_correct.ravel())
    score_correct = model.evaluate(X_test, y_correct, verbose=0)

    #predicted age difference (corrected)
    pad_c = y_correct - y_test
    print(f'PAD_c for TD (test set) = {pad_c.mean()} (std {pad_c.std()})')

    plt.scatter(y_test,y_correct, color='green', alpha=0.7, label=f'r={r_td_correct:.2}\nMAE = {score_correct:.3} years\nPAD = {pad_c.mean():.2} years')
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

    r_asd, _ = pearsonr(y_asd,y_pred_asd.ravel())

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
    r_asd_correct, _ = pearsonr(y_asd,y_correct_asd.ravel())
    score_correct_asd = model.evaluate(X_asd, y_correct_asd, verbose=0)
    pad_c = ((y_pred_asd.ravel()-b)/a) - y_asd
    print(f'PAD_c for ASD = {pad_c.mean()} (std = {pad_c.std()})')
    plt.xlabel('Chronological Age [years]')
    plt.ylabel('Predicted Age [years]')
    plt.legend()

    plt.subplot(122)
    plt.title('ASD (Corrected)')

    plt.scatter(y_asd,y_correct_asd, color='purple', alpha=0.7, label=f'r={r_asd_correct:.2}\nMAE = {score_correct_asd:.3} years\nPAD = {pad_c.mean():.2} years')
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

    plt.show()




if __name__ == "__main__":
    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    if 1:
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