# MODEL CONFIG
import config

import numpy as np

# DEEP LEARNING
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


import folder_class
import keras_tuner
import joblib 
import Model_Class
import os

class Train_Model ():
    def __init__(self, X_data_reshaped, X_shift_reshaped, y_data_scaled):
        self.variables (X_data_reshaped, X_shift_reshaped, y_data_scaled)       

    def variables (self, X_data_reshaped, X_shift_reshaped, y_data_scaled):
        self.X_data_reshaped = X_data_reshaped
        self.X_shift_reshaped = X_shift_reshaped       
        self.y_data_scaled = y_data_scaled
        self.epochs = config.epochs
        self.batch_size = config.batch_size
    
    def Optimizer (self):

        Keras_Tuner_path = folder_class.folder.create("Keras_Tuner_Optimizer")

        early_stopping = EarlyStopping(monitor='val_loss', patience = 2)
        
        print ("\n\nEXECUTE MODEL OPTIMIZATION\n\n")

        lstm = Model_Class.LSTM_Model (self.X_data_reshaped, self.y_data_scaled)
        lstm.variables (self.X_data_reshaped, self.y_data_scaled)

        lstm.build_model (keras_tuner.HyperParameters())

        tuner = keras_tuner.RandomSearch(
            hypermodel = lstm.build_model,
            objective = "val_loss",
            max_trials = 5,
            executions_per_trial = 2,
            overwrite = True,
            directory = Keras_Tuner_path,
            project_name = "Optimizer",
        )
        
        tuner.search_space_summary()
        print ("\n\n")
        tuner.search(self.X_data_reshaped, self.y_data_scaled, epochs = self.epochs, batch_size = self.batch_size, validation_split=0.1, callbacks = [early_stopping], verbose = 1) 
        print ("\n\n")
        tuner.results_summary()
        print ("\n\n")
        tuner.get_best_hyperparameters()
        print ("\n\n")
        tuner.get_best_models()

        best_hyperparameters = tuner.get_best_hyperparameters(5)

        print (best_hyperparameters)
    
    def Full_Train (self):
        
        Keras_Tuner_path = os.path.abspath("Keras_Tuner_Optimizer")        
        data_scaling = os.path.abspath ("data_scaling")
        y_path = os.path.join(data_scaling, 'Model_Scaler_y.gz')
        y_scaler = joblib.load(y_path)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience = 2)

        print ("\n\nEXECUTE FULL MODEL TRAINING\n\n")

        lstm = Model_Class.LSTM_Model (self.X_data_reshaped, self.y_data_scaled)
        lstm.variables (self.X_data_reshaped, self.y_data_scaled)

        lstm.build_model (keras_tuner.HyperParameters())

        tuner = keras_tuner.RandomSearch(
                hypermodel = lstm.build_model,
                objective = "val_loss",
                max_trials = 5,
                executions_per_trial = 2,
                overwrite = False,
                directory = Keras_Tuner_path,
                project_name = "Optimizer",
            )

        best_hyperparameters = tuner.get_best_hyperparameters(5)

        final_model = lstm.build_model (best_hyperparameters[0])
        
        # Fitting the RNN to the Training set
        final_model.fit(self.X_data_reshaped, self.y_data_scaled, epochs = self.epochs, batch_size = self.batch_size, validation_split=0.1, callbacks = [early_stopping])
        final_model.summary()

        # Save Model
        model_output = folder_class.folder.create("Model_Output_Folder")

        final_model.save(model_output)
        final_model.save_weights(model_output)
        
        predictions = final_model.predict(self.X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)

        print ("FINAL PREDICTIONS \n", predictions)

    def Re_Train (self):
        
        early_stopping = EarlyStopping(monitor='val_loss', patience = 2)

        print ("\n\nEXECUTE RE-TRAINING\n\n")

        # Path for Model Outputs
        Model_Output_Folder = os.path.abspath ("Model_Output_Folder")

        # Loads previously saved model
        final_model = load_model(Model_Output_Folder)
        final_model.summary()

        # RE-FITS NEW DATA
        final_model.fit(self.X_data_reshaped, self.y_data_scaled, epochs = self.epochs, batch_size = self.batch_size, validation_split=0.1, callbacks = [early_stopping])

        # Save Model
        final_model.save(Model_Output_Folder)
        final_model.save_weights(Model_Output_Folder)
        
        # Path for Data Scaler 
        data_scaling = os.path.abspath ("data_scaling")
        y_path = os.path.join(data_scaling, 'Model_Scaler_y.gz')
        y_scaler = joblib.load(y_path)

        # Predict

        predictions = final_model.predict(self.X_shift_reshaped)
        predictions = np.squeeze(predictions)
        predictions = y_scaler.inverse_transform(predictions)
    

        print ("FINAL PREDICTIONS \n", predictions)