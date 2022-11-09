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

        #CREATE LONG SHORT TERM MEMORY RECURRENT NEURAL NETWORK
        model = Sequential()
        #Adding the first LSTM layer 
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_data_reshaped.shape[1], X_data_reshaped.shape[2])))

        model.summary()
        # Droupout Layer 
        # Second LSTM layer 
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))  
        # Third LSTM layer 
        model.add(LSTM(units = 50, return_sequences = True))
        # Adding a fourth LSTM layer 
        model.add(LSTM(units = 50))
        # Adding the output layer
        model.add(Dense(units = y_data_scaled.shape[1], activation = 'relu'))  # relu function keeps the model from giving negative outputs
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

        # Fitting the RNN to the Training set
        model.fit(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size, validation_split=0.1, callbacks = [early_stopping])
        model.summary()

        # Save Model
        model.save(model_path)
        model.save_weights(model_weights_path)
        
        predictions = model.predict(X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)
    
    elif full_training is False:

        print ("EXECUTE RE-TRAINING")
        
        #Loads previously saved model
        model = load_model(model_path)
        model.summary()

        #RE-FITS NEW DATA
        model.fit(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size)
        predictions = model.predict(X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)

    else:
        print("Error: Full Training variable is invalid")
        quit()
    
    return predictions