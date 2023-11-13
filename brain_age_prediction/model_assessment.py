from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import pickle
import matplotlib.pyplot as plt

SEED = 7 #for reproducibility

def plot_loss(history, structural=False, functional=False):
    '''
    plot the loss during training
    '''
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    if structural:
        plt.title('Structural Model')
    if functional:
        plt.title('Functional Model')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')

    if structural:
        plt.savefig('plots/loss_structural_model.pdf')
    if functional:
        plt.savefig('plots/loss_functional_model.pdf')
    plt.show()


def retrain(X_train,y_train,X_test,y_test,functional=False,structural=False):
    '''
    Re-train the best model obtained throug model selection
    on all the available data (of the training set). Then evaluate
    the model on the test set.
    Finally save the model to file
    '''

    if structural:
        filename = 'structural_model_hyperparams.pkl'
        create_model = create_structural_model

    if functional:
        filename = 'functional_model_hyperparams.pkl'
        create_model = create_functional_model

    # Read dictionary pkl file
    with open(os.path.join('best_hyperparams',filename), 'rb') as fp:
        best_hyperparams = pickle.load(fp)


    model = create_model(dropout=best_hyperparams['model__dropout'],
                         hidden_neurons=best_hyperparams['model__hidden_neurons'],
                         hidden_layers=best_hyperparams['model__hidden_layers'])

    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    max_epochs = 1000

    train = model.fit(X_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=32,
                    verbose=1,
                    validation_data=(X_test,y_test),
                    callbacks=[reduce_lr,early_stopping])

    #evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'TEST MAE = {score}')

    #save model to disk
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

    plot_loss(history=train.history,structural=structural,functional=functional)


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split

    #structural model
    print('--------STRUCTURAL MODEL---------')

    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

    X = preprocessing(df_s_td)
    y = np.array(df_s_td['AGE_AT_SCAN'])

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=SEED)

    retrain(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,structural=True)



    #functional model
    print('--------FUNCTIONAL MODEL---------')

    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    X = preprocessing(df_f_td)
    y = np.array(df_f_td['AGE_AT_SCAN'])

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=SEED)
    retrain(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,functional=True)
