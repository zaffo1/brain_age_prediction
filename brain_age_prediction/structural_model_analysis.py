from useful_functions import load_dataset, preprocessing, line



if __name__ == "__main__":
    import numpy as np
    import os

    DATASET_NAME = 'Harmonized_structural_features.csv'
    df_s_td, df_s_asd = load_dataset(dataset_name=DATASET_NAME)

    #inputs
    X = preprocessing(df_s_td)
    #targets
    y = np.array(df_s_td['AGE_AT_SCAN'])

    from sklearn.model_selection import train_test_split
    # shuffle and split training and test sets
    SEED = 7 #for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=SEED)

    from tensorflow.keras.models import model_from_json
    # load json and create model
    json_file = open(os.path.join('saved_models','structural_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join('saved_models','structural_model_weights.h5'))

    from keras.optimizers.legacy import Adam

    optim = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optim)
    print("Loaded model from disk")

    import matplotlib.pyplot as plt

    plt.figure(1)
    y_pred = model.predict(X_test)
    plt.scatter(y_test,y_pred)
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')

    from scipy.optimize import curve_fit

    xx = y_test
    yy = y_pred.ravel()
    popt, pcov = curve_fit(line, xx, yy)
    a, b = popt
    print(popt)
    plt.scatter(y_test,y_pred)
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.plot(x,line(x,a,b), color = 'red', linestyle='--')


    pad_c = ((y_pred.ravel()-b)/a) - y_test
    print(f'PAD for TD (test set) {pad_c.mean()}')


    #ASD
    X_asd = preprocessing(df_s_asd)
    y_asd = np.array(df_s_asd['AGE_AT_SCAN'])
    y_pred_asd = model.predict(X_asd)

    from scipy.stats import pearsonr

    r_td, _ = pearsonr(y_test,y_pred.ravel())
    r_asd, _ = pearsonr(y_asd,y_pred_asd.ravel())

    plt.figure(2, figsize=(20,10))
    plt.scatter(y_test,y_pred, color='blue', alpha=0.7, label=f'TD, r={r_td:.2}')
    plt.scatter(y_asd,y_pred_asd, color='red', alpha =0.5, label=f'ASD, r={r_asd:.2}')
    #plt.xlim(6,20)
    #plt.ylim(5,25)
    plt.title('Structural Model')
    plt.xlabel('Actual Age [years]')
    plt.ylabel('Predicted Age [years]')

    x = np.linspace(min(y_asd),max(y_asd),1000)
    plt.plot(x,x, color = 'grey', linestyle='--',label='perfect regression')
    plt.legend()
    plt.savefig('plots/age_regression_structural_model.pdf')


    pad_c = ((y_pred_asd.ravel()-b)/a) - y_asd
    print(pad_c.mean())
    print(pad_c.std())
    plt.show()


