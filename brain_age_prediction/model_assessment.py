from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import pickle
def retrain(X_train,y_train,X_test,y_test,functional=False,structural=False):
     #re-train the model using all available data.

    if structural:
        filename = 'structural_model_hyperparams.pkl'
        create_model = create_structural_model

    if functional:
        filename = 'functional_model_hyperparams.pkl'
        create_model = create_functional_model

    # Read dictionary pkl file
    with open(os.path.join('best_hyperparams',filename), 'rb') as fp:
        best_hyperparams = pickle.load(fp)


    model = create_model(input_neurons=best_hyperparams['model__input_neurons'],
                          hidden_neurons=best_hyperparams['model__hidden_neurons'],
                          hidden_layers=best_hyperparams['model__hidden_layers'])
    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    MAX_EPOCHS = 1000
    train=model.fit(X_train,
                    y_train,
                    epochs=MAX_EPOCHS,
                    batch_size=32,
                    verbose=1,
                    validation_data=(X_test,y_test),
                    callbacks=[reduce_lr,early_stopping])

    #evalueate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'TEST MAE = {score}')

    if structural:
        json_name = 'structural_model.json'
        h5_name = 'structural_model_weights.h5'

    if functional:
        json_name = 'functional_model.json'
        h5_name = 'functional_model_weights.h5'

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models',json_name), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join('saved_models',h5_name))
    print("Saved model to disk")


    # plot loss during training
    import matplotlib.pyplot as plt
    plt.plot(train.history['loss'], label='train')
    plt.plot(train.history['val_loss'], label='test')
    if structural:
        plt.title('Structural Model')
    if functional:
        plt.title('Functional Model')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    SEED = 7 #for reproducibility

    if 1:
        #structural model
        print('--------STRUCTURAL MODEL---------')

        df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

        X = preprocessing(df_s_td)
        y = np.array(df_s_td['AGE_AT_SCAN'])

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=SEED)


        retrain(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,structural=True)


    if 1:
        #functional model
        print('--------FUNCTIONAL MODEL---------')

        df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

        X = preprocessing(df_f_td)
        y = np.array(df_f_td['AGE_AT_SCAN'])

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)
        retrain(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,functional=True)
