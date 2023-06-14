# DEEP LEARNING

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

class LSTM_Model ():
    def __init__(self, X_data_reshaped, y_data_scaled):
        self.variables (X_data_reshaped, y_data_scaled)       

    def variables (self, X_data_reshaped, y_data_scaled):
        self.X_data_reshaped = X_data_reshaped
        self.y_data_scaled = y_data_scaled       

    def build_model (self, hp):

        X1 = self.X_data_reshaped.shape[1]
        X2 = self.X_data_reshaped.shape[2]
        y1 = self.y_data_scaled.shape[1]

        # KERAS TUNER OPTIMIZER for the hyperparameters: Extra Layers, Units, Adam Optimizer Learning Rate, Droupout

        units_tuner = hp.Int("units", min_value=1, max_value=100)
        dropout_tuner = hp.Float("dropout", min_value=0.05, max_value=0.5)
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        layers_tuner = hp.Int("num_layers", 0, 3)
        
        # CREATE LONG SHORT TERM MEMORY RECURRENT NEURAL NETWORK          
        model = Sequential()

        # LSTM FIRST LAYER 
        model.add(LSTM(units = units_tuner, return_sequences = True, input_shape = (X1, X2)))

        # TUNE EXTRA LAYERS:
        for i in range(layers_tuner):
            model.add(LSTM(units = units_tuner, return_sequences = True))

        # FINAL LSTM LAYER: without "return_sequences = True"
        model.add(LSTM(units =  units_tuner))

        # DROPOUT LAYER
        model.add(Dropout(dropout_tuner))  

        # DENSE OUTPUT LAYER
        model.add(Dense(units = y1, activation = 'relu'))

        # Compiling the RNN 
        model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'mean_squared_error', metrics = 'mean_squared_error')

        return model    