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

    #re-train the model using all available data.
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping

    import pickle

    # Read dictionary pkl file
    with open(os.path.join('best_hyperparams','structural_model_hyperparams.pkl'), 'rb') as fp:
        best_hyperparams = pickle.load(fp)

    model = create_model(input_neurons=best_hyperparams['model__input_neurons'],
                          hidden_neurons=best_hyperparams['model__hidden_neurons'],
                          hidden_layers=best_hyperparams['model__hidden_layers'])
    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    MAX_EPOCHS = 500
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

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models','structural_model.json'), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join('saved_models','structural_model_weights.h5'))
    print("Saved model to disk")


    # plot loss during training
    import matplotlib.pyplot as plt
    plt.plot(train.history['loss'], label='train')
    plt.plot(train.history['val_loss'], label='test')
    plt.title('Model Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')
    plt.show()