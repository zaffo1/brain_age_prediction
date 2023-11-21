'''
Perform Model Selection for the joint model
'''
import time
import datetime
from pathlib import Path
from brain_age_prediction.utils.loading_data import load_train_test
from brain_age_prediction.utils.model_selection_utils import model_selection

ROOT_PATH = Path(__file__).parent.parent
SEED = 7 #fixed for reproducibility



if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np

    #check if GPU is available
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    start_time = time.time()

    #load data
    X_s_train, X_s_test, y_s_train, y_s_test, X_f_train, X_f_test, y_f_train, y_f_test  =(
        load_train_test(split=0.3,seed=SEED))

    #joint model grid search
    print('--------JOINT MODEL--------')

    y = y_f_train
    merged_inputs = np.concatenate([X_f_train,X_s_train], axis=-1)


    dropout = [0.1,0.2,0.5]
    hidden_neurons = [50,100,150,200]
    hidden_layers = [1,2,3]
    search = [dropout, hidden_neurons, hidden_layers]

    model_selection(search_space=search, x_train=merged_inputs, y_train=y,model_type='joint')
    tf.keras.backend.clear_session()

    elapsed_seconds = time.time() - start_time
    print('Finished model selection...'
           f'Elapsed time: {str(datetime.timedelta(seconds=elapsed_seconds))} (h:m:s)')
