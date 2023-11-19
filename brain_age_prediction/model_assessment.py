'''
Retrain the selected model, and assess their performances.
'''

import sys
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from brain_age_prediction.utils.loading_data import load_train_test
from brain_age_prediction.utils.custom_models import (create_functional_model,
                                                      create_structural_model,
                                                      create_joint_model)
from brain_age_prediction.utils.chek_model_type import check_model_type

ROOT_PATH = Path(__file__).parent.parent
SEED = 7 #for reproducibility

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title



def load_model_architecture(model_type):
    '''
    Depending on the type of model given in input,
    load it with the best hyperparameters found and
    return the loaded model itself
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    filename = f'{model_type}_model_hyperparams.pkl'

    if model_type == 'structural':
        create_model = create_structural_model
    if model_type == 'functional':
        create_model = create_functional_model
    if model_type == 'joint':
        create_model = create_joint_model

    # Read dictionary pkl file
    try:
        with open(os.path.join(
            ROOT_PATH,'brain_age_prediction','best_hyperparams',filename), 'rb') as fp:
            best_hyperparams = pickle.load(fp)
    except OSError as e:
        print('Cannot load best hyperparameters:'
               f'cannot read the file in which they should be saved! \n{e}')
        sys.exit(1)

    model = create_model(dropout=best_hyperparams['model__dropout'],
                         hidden_neurons=best_hyperparams['model__hidden_neurons'],
                         hidden_layers=best_hyperparams['model__hidden_layers'])

    return model

def save_model(model,model_type):
    '''
    save model to disk
    takes in input the model
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    json_name = f'{model_type}_model.json'
    h5_name = f'{model_type}_model_weights.h5'

    # serialize model to JSON
    model_json = model.to_json()

    try:
        with open(os.path.join(ROOT_PATH,'brain_age_prediction','saved_models',json_name),
                   'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5

        model.save_weights(os.path.join(ROOT_PATH,'brain_age_prediction','saved_models',h5_name))
        print("Saved model to disk")
    except OSError as e:
        print(f'Cannot save the model! \n{e}')
        sys.exit(1)


def plot_loss(history,loss, model_type):
    '''
    plot the loss during training, and save training curves to file
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    plt.figure(f'Loss {model_type} model',figsize=[8,6])
    plt.plot(history['loss'], label='Train', color='black')
    plt.plot(history['val_loss'], label='Test', color='red', linestyle='--')

    plt.title(f'{model_type.capitalize()} Model (Test MAE = {loss:.3} years)')

    plt.xlabel('Epochs')
    plt.ylabel('Loss Values')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(
            ROOT_PATH,'brain_age_prediction','plots',f'loss_{model_type}_model.pdf'))

    plt.show()


def retrain(x_train,y_train,x_test,y_test,model_type):
    '''
    Re-train the best model obtained throug model selection
    on all the available data (of the training set). Then evaluate
    the model on the test set.
    Finally save the model to file
    '''
    try:
        check_model_type(model_type)
    except AssertionError as e:
        print(e)
        sys.exit(1)


    model = load_model_architecture(model_type)

    plot_model(model, os.path.join(ROOT_PATH,'brain_age_prediction',
                        'plots',f'architecture_{model_type}_model.png'), show_shapes=True)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=10, min_lr=0.00001)

    max_epochs = 200

    train = model.fit(x_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=64,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[reduce_lr])

    #evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'TEST MAE = {score}')
    #save model to disk
    save_model(model,model_type)
    plot_loss(history=train.history, loss=score, model_type=model_type)


if __name__ == "__main__":
    import tensorflow as tf

    #check if GPU is available
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #load data
    x_s_train, x_s_test, y_s_train, y_s_test,x_f_train, x_f_test, y_f_train, y_f_test  = (
        load_train_test(split=0.3,seed=SEED))

    #structural model
    print('--------STRUCTURAL MODEL---------')
    retrain(x_train=x_s_train,y_train=y_s_train,
            x_test=x_s_test,y_test=y_s_test,model_type='structural')

    #functional model
    print('--------FUNCTIONAL MODEL---------')
    retrain(x_train=x_f_train,y_train=y_f_train,
            x_test=x_f_test,y_test=y_f_test,model_type='functional')

    #joint model
    print('--------JOINT MODEL---------')

    #check if y_f_... == y_s_...

    retrain(x_train=[x_f_train,x_s_train],y_train=y_f_train,x_test=[x_f_test,x_s_test],
            y_test=y_f_test,model_type='joint')
