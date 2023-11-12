from useful_functions import load_dataset, preprocessing, line
from tensorflow.keras.models import model_from_json
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_joint_model():
    '''
    Load the saved keras joint model, and compile it
    return the compiled model
    '''


    # load json and create model
    json_file         = open(os.path.join('saved_models','joint_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join('saved_models','joint_model_weights.h5'))

    optim = Adam(learning_rate = 0.001)
    model.compile(loss='mae', optimizer=optim)
    print("Loaded model from disk")
    return model


def model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd):
    '''
    function that performs an analysis of the results obtained using the
    regression models obtained.
    In particular: ..........
    '''
    X_s = preprocessing(df_s_td)
    X_f = preprocessing(df_f_td)

    #check if targets are equal
    y_s = df_s_td['AGE_AT_SCAN']
    y_f = df_f_td['AGE_AT_SCAN']
    print(y_s.equals(y_f))
    y = np.array(y_s)

    SEED = 7
    X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test = train_test_split(
        X_f,X_s, y, test_size=0.3, random_state=SEED)

    #evaluate model
    score = model.evaluate([X_f_test,X_s_test], y_test, verbose=0)
    print(f'TEST MAE = {score}')

    y_pred = model.predict([X_f_test,X_s_test])
    plt.figure(1)
    plt.scatter(y_test,y_pred)
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')

    popt, pcov = curve_fit(line, y_test, y_pred.ravel())
    a, b = popt
    print(f'a = {popt[0]}, b={popt[1]}')

    plt.scatter(y_test,y_pred, color='b')
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.plot(x,line(x,a,b), color = 'red', linestyle='--')

    #predicted age difference (corrected)
    pad_c = ((y_pred.ravel()-b)/a) - y_test
    print(f'PAD_c for TD (test set) = {pad_c.mean()} (std {pad_c.std()})')


    #ASD
    X_f_asd = preprocessing(df_f_asd)
    X_s_asd = preprocessing(df_s_asd)

    y_asd = np.array(df_f_asd['AGE_AT_SCAN'])

    y_pred_asd = model.predict([X_f_asd,X_s_asd])

    r_td, _ = pearsonr(y_test,y_pred.ravel())
    r_asd, _ = pearsonr(y_asd,y_pred_asd.ravel())

    plt.figure(2)
    plt.scatter(y_test,y_pred, color='blue', alpha=0.7, label=f'TD, r={r_td:.2}')
    plt.scatter(y_asd,y_pred_asd, color='red', alpha =0.5, label=f'ASD, r={r_asd:.2}')
    plt.title('Joint Model')
    plt.xlabel('Actual Age [years]')
    plt.ylabel('Predicted Age [years]')

    x = np.linspace(min(y_asd),max(y_asd),1000)
    plt.plot(x,x, color = 'grey', linestyle='--',label='perfect regression')
    plt.legend()
    plt.savefig('plots/age_regression_joint_model.pdf')


    pad_c = ((y_pred_asd.ravel()-b)/a) - y_asd
    print(f'PAD_c for ASD = {pad_c.mean()} (std = {pad_c.std()})')
    plt.show()



if __name__ == "__main__":


    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    model             = load_joint_model()

    model_analysis(model,df_s_td,df_s_asd,df_f_td,df_f_asd)
