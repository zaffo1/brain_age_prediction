from useful_functions import load_dataset, preprocessing, create_functional_model, create_structural_model, create_joint_model
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import os

def train(X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test):
    '''
    Train the joint model
    '''

    model = create_joint_model(30,1)

    #plot the model architecture
    plot_model(model, "architecture_joint_model.png", show_shapes=True)
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    MAX_EPOCHS = 1000

    train=model.fit([X_f_train,X_s_train],
                    y_train,
                    epochs=MAX_EPOCHS,
                    batch_size=32,
                    verbose=1,
                    validation_data=([X_f_test,X_s_test],y_test),
                    callbacks=[reduce_lr,early_stopping])

    #evalueate model
    score = model.evaluate([X_f_test,X_s_test], y_test, verbose=0)
    print(f'TEST MAE = {score}')

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models','joint_model.json'), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join('saved_models','joint_model_weights.h5'))
    print("Saved model to disk")

    # plot loss during training
    plt.plot(train.history['loss'], label='train')
    plt.plot(train.history['val_loss'], label='test')
    plt.title('Joint Model')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping

    SEED = 7

    df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')
    df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

    X_s = preprocessing(df_s_td)
    X_f = preprocessing(df_f_td)

    #check if targets are equal
    y_s = df_s_td['AGE_AT_SCAN']
    y_f = df_f_td['AGE_AT_SCAN']
    print(y_s.equals(y_f))
    y = np.array(y_s)

    X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test = train_test_split(
        X_f,X_s, y, test_size=0.3, random_state=SEED)

    train(X_f_train, X_f_test, X_s_train, X_s_test, y_train, y_test)
