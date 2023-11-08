from useful_functions import load_dataset, preprocessing, create_model



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

    y_pred = model.predict(X_test)
    plt.scatter(y_test,y_pred)
    x = np.linspace(min(y_test),max(y_test),1000)
    plt.plot(x,x, color = 'grey', linestyle='--')
    plt.show()