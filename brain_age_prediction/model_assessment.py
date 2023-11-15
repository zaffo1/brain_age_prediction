import sys
import os
import pickle
import matplotlib.pyplot as plt
from useful_functions import load_train_test, create_functional_model, create_structural_model, create_joint_model
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model


SEED = 7 #for reproducibility

def plot_loss(history,loss, structural=False, functional=False, joint=False):
    '''
    plot the loss during training, and save training curves to file
    '''
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    if structural:
        plt.title(f'Structural Model (Test MAE = {loss:.3} years)')
    if functional:
        plt.title(f'Functional Model (Test MAE = {loss:.3} years)')
    if joint:
        plt.title(f'Joint Model (Test MAE = {loss:.3} years)')

    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')

    if structural:
        plt.savefig('plots/loss_structural_model.pdf')
    if functional:
        plt.savefig('plots/loss_functional_model.pdf')
    if joint:
        plt.savefig('plots/loss_joint_model.pdf')

    plt.show()


def retrain(X_train,y_train,X_test,y_test,functional=False,structural=False,joint=False):
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

    if joint:
        filename = 'joint_model_hyperparams.pkl'
        create_model = create_joint_model

    # Read dictionary pkl file
    try:
        with open(os.path.join('best_hyperparams',filename), 'rb') as fp:
            best_hyperparams = pickle.load(fp)
    except OSError as e:
        print(f'Cannot load best hyperparameters: cannot read the file in which they should be saved! \n{e}')
        sys.exit(1)


    model = create_model(dropout=best_hyperparams['model__dropout'],
                         hidden_neurons=best_hyperparams['model__hidden_neurons'],
                         hidden_layers=best_hyperparams['model__hidden_layers'])

    if structural:
        plot_model(model, "plots/architecture_structural_model.png", show_shapes=True)

    if functional:
        plot_model(model, "plots/architecture_functional_model.png", show_shapes=True)

    if joint:
        plot_model(model, "plots/architecture_joint_model.png", show_shapes=True)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=10, min_lr=0.00001)

    if structural:
        max_epochs = 100
    if functional or joint:
        max_epochs = 300

    train = model.fit(X_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=32,
                    verbose=1,
                    validation_data=(X_test,y_test),
                    callbacks=[reduce_lr])

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

    if joint:
        json_name = 'joint_model.json'
        h5_name = 'joint_model_weights.h5'

    # serialize model to JSON
    model_json = model.to_json()

    try:
        with open(os.path.join('saved_models',json_name), 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join('saved_models',h5_name))
        print("Saved model to disk")
    except OSError as e:
        print(f'Cannot save the model! \n{e}')
        sys.exit(1)

    plot_loss(history=train.history, loss=score,structural=structural,functional=functional,joint=joint)


if __name__ == "__main__":
    import numpy as np

    #load data
    X_s_train, X_s_test, y_s_train, y_s_test,X_f_train, X_f_test, y_f_train, y_f_test  = load_train_test(split=0.3,seed=SEED)

    if 1:
        #structural model
        print('--------STRUCTURAL MODEL---------')
        retrain(X_train=X_s_train,y_train=y_s_train,X_test=X_s_test,y_test=y_s_test,structural=True)

        #functional model
        print('--------FUNCTIONAL MODEL---------')
        retrain(X_train=X_f_train,y_train=y_f_train,X_test=X_f_test,y_test=y_f_test,functional=True)

    #joint model
    print('--------JOINT MODEL---------')

    y_test = y_f_test
    y_train = y_f_train

    retrain(X_train=[X_f_train,X_s_train],y_train=y_train,X_test=[X_f_test,X_s_test],y_test=y_test,joint=True)
