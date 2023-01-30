# MODEL CONFIG
import config as config
import numpy as np

# DEEP LEARNING MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import keras_tuner

from sklearn.metrics import mean_squared_error

if __name__ == "__model_training__":
    model_training ()

if __name__ == "__tuner_optimizer__":
    tuner_optimizer ()

def model_training (X_data_reshaped,
                    y_data_scaled):


    epochs = config.epochs
    batch_size = config.batch_size
    Keras_Tuner_path = config.Keras_Tuner_path
    model_path = config.model_path
    model_weights_path = config.model_weights_path

    early_stopping = EarlyStopping(monitor='val_loss', patience = 2)

    print ("\n\nEXECUTE FULL TRAINING\n\n")
    
    print ("X_data_reshaped\n", X_data_reshaped)
    print ("y_data_scaled\n", y_data_scaled)
    print ("\n\nX_data_reshaped Shape: ", X_data_reshaped.shape)
    print ("y_data_scaled Shape: ", y_data_scaled.shape)
    
    def build_model (hp):            

            # KERAS TUNER OPTIMIZER for the hyperparameters: Extra Layers, Units, Adam Optimizer Learning Rate, Droupout

            units_tuner = hp.Int("units", min_value=1, max_value=100)
            dropout_tuner = hp.Float("dropout", min_value=0.05, max_value=0.5)
            learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
            layers_tuner = hp.Int("num_layers", 0, 3)
            
            # CREATE LONG SHORT TERM MEMORY RECURRENT NEURAL NETWORK          
            model = Sequential()

            # LSTM FIRST LAYER 
            model.add(LSTM(units = units_tuner, return_sequences = True, input_shape = (X_data_reshaped.shape[1], X_data_reshaped.shape[2])))
            
            # TUNE EXTRA LAYERS:
            for i in range(layers_tuner):
                model.add(LSTM(units = units_tuner, return_sequences = True))

            # FINAL LSTM LAYER: without "return_sequences = True"
            model.add(LSTM(units =  units_tuner))

            # DROPOUT LAYER
            model.add(Dropout(dropout_tuner))  

            # DENSE OUTPUT LAYER
            model.add(Dense(units = y_data_scaled.shape[1], activation = 'relu'))
            
            # Compiling the RNN 
            model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'mean_squared_error', metrics = 'mean_squared_error')
            
            return model

    def tuner_optimizer ():
        
        build_model (keras_tuner.HyperParameters())

        tuner = keras_tuner.RandomSearch(
            hypermodel = build_model,
            objective = "val_loss",
            max_trials = 5,
            executions_per_trial = 2,
            overwrite = True,
            directory = Keras_Tuner_path,
            project_name = "Optimizer",
        )
        
        tuner.search_space_summary()
        print ("\n\n")
        #tuner.search(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size, validation_data = (X_test_reshaped, y_test_scaled), callbacks = [early_stopping], verbose = 1) 
        tuner.search(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size, callbacks = [early_stopping], verbose = 1) 
        print ("\n\n")
        tuner.results_summary()
        print ("\n\n")
        tuner.get_best_hyperparameters()
        print ("\n\n")
        tuner.get_best_models()

        best_hyperparameters = tuner.get_best_hyperparameters(5)

        return best_hyperparameters

    best_hyperparameters = tuner_optimizer ()

    return best_hyperparameters