import numpy as np

# DEEP LEARNING MODEL
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner

from sklearn.metrics import mean_squared_error

if __name__ == "__model_training__":
    model_training ()

def model_training (full_training,
                    X_data_reshaped,
                    y_data_scaled,
                    epochs,
                    batch_size,
                    X_shift_reshaped,
                    model_path,
                    model_weights_path,
                    y_scaler):

    early_stopping = EarlyStopping(monitor='val_loss', patience = 2)

    if full_training is True:

        print ("\n\nEXECUTE FULL TRAINING\n\n")
     
        print ("X_data_reshaped\n", X_data_reshaped)
        print ("y_data_scaled\n", y_data_scaled)
        print ("\n\nX_data_reshaped Shape: ", X_data_reshaped.shape)
        print ("y_data_scaled Shape: ", y_data_scaled.shape)
        print ("Batch Size: ", batch_size)

        def build_model (hp):

            #CREATE LONG SHORT TERM MEMORY RECURRENT NEURAL NETWORK

            units_tuner = hp.Int("units", min_value=1, max_value=100)
            dropout_tuner = hp.Float("dropout", min_value=0.05, max_value=0.5)

            model = Sequential()
            #Adding the first LSTM layer 
            model.add(LSTM(units = units_tuner, return_sequences = True, input_shape = (X_data_reshaped.shape[1], X_data_reshaped.shape[2])))
            # Droupout Layer 
            model.add(Dropout(dropout_tuner))  

            # Tune the number of extra LSTM layers.
            for i in range(hp.Int("num_layers", 1, 3)):
                model.add(LSTM(units = units_tuner, return_sequences = True))

            # Adding a fourth LSTM layer 
            model.add(LSTM(units =  units_tuner))
            # Adding the output layer
            model.add(Dense(units = y_data_scaled.shape[1], activation = 'relu'))  # relu function keeps the model from giving negative outputs
            # Compiling the RNN
            model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
            
            return model

        build_model (keras_tuner.HyperParameters())

        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=5,
            executions_per_trial=2,
            overwrite=True,
            directory="/Users/telmopaiva/Documents/ML_Python/LSTM_Sequence_Forecast/Keras_Tuner_Optimizer",
            project_name="test",
        )
        
        tuner.search_space_summary()
        print ("\n\n")
        tuner.search(X_data_reshaped, y_data_scaled, epochs = 10, batch_size = 1, validation_split=0.1, callbacks = [early_stopping]) 
        print ("\n\n")
        tuner.results_summary()
        print ("\n\n")
        tuner.get_best_hyperparameters()
        print ("\n\n")
        tuner.get_best_models()

        best_hyperparameters = tuner.get_best_hyperparameters(5)    

        final_model = build_model (best_hyperparameters[0])
        
        # Fitting the RNN to the Training set
        final_model.fit(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size, validation_split=0.1, callbacks = [early_stopping])
        final_model.summary()

        # Save Model
        final_model.save(model_path)
        final_model.save_weights(model_weights_path)
        
        predictions = final_model.predict(X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)
    
    elif full_training is False:

        print ("EXECUTE RE-TRAINING")
        
        #Loads previously saved model
        final_model = load_model(model_path)
        final_model.summary()

        #RE-FITS NEW DATA
        final_model.fit(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size)
        predictions = final_model.predict(X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)

    else:
        print("Error: Full Training variable is invalid")
        quit()
    
    return predictions