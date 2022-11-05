###################################################
#                   MAIN FILE                     #
###################################################
###################################################
#            Tablua Rasa RUNS FROM HERE             #
###################################################

# MODEL CONFIG
import ModelConfig as config

# THE USUAL SUSPECTS
import pandas as pd
import numpy as np
import copy
import openpyxl

# SSH TUNNEL
from sshtunnel import SSHTunnelForwarder

# CUSTOM FUNCTIONS
import DBQueryFunctions  
import LSTMhelpFunctions 
import PlotGenerator
import Model

# SSH TUNNEL TO BARE METAL SERVER
#with SSHTunnelForwarder(  
#    config.IP_Address,
#    ssh_username = config.username,
#    ssh_pkey = config.PEM_Location,
#    remote_bind_address = (config.bind_address, config.port),
#    local_bind_address = (config.bind_address, config.port)):
    
#    print ('SSH CONNECTION STARTED')

#DB QUERIES AND DATASET CREATION

ListStations = DBQueryFunctions.GetListStation (config.station, 
                                                config.limit)

X_data, y_data = DBQueryFunctions.MergeStationsData (ListStations, 
                                                    config.full_training,
                                                    config.y_inputs, 
                                                    config.X_inputs,
                                                    config.data_intervals,
                                                    config.last_dataset_row_path,
                                                    config.last_retrain_dataset_row_path)


#pd.set_option('display.max_rows', None)
print ("X_data\n", X_data)
print ("y_data\n", y_data)
print ("y_data shape: ", y_data.shape)
print ("X_data shape: ", X_data.shape)

# FUTURE TIME SERIES SEQUENCE
X_shift_data = LSTMhelpFunctions.future_time_series (config.hours,
                                                    config.data_intervals, 
                                                    config.X_inputs)

X_shift_data = LSTMhelpFunctions.extractTimeInfo(config.X_inputs, X_shift_data)

num_data = len(X_data) - len(X_shift_data)
y_test_data = y_data[num_data:] #REAL DATA FOR PLOTTING

#pd.set_option('display.max_rows', None) 
print ("y_test_data", y_test_data)
print("y_test_data shape: ", y_test_data.shape)
print ("X_shift_data", X_shift_data)
print ("X_shift_data shape: ", X_shift_data.shape)

# FEATURE SCALLING
X_data_scaled, X_shift_scaled, y_data_scaled, y_scaler = LSTMhelpFunctions.data_scalling    (config.full_training, 
                                                                                            X_data, 
                                                                                            X_shift_data, 
                                                                                            y_data,
                                                                                            config.X_scaler_path, 
                                                                                            config.y_scaler_path)

# DATA INPUT RESHAPING 
X_data_reshaped = X_data_scaled.reshape (X_data.shape[0], 1, X_data.shape[1])

X_shift_reshaped = X_shift_scaled.reshape (X_shift_data.shape[0], 1, X_shift_data.shape[1])

# CREATE PREDICTION HEADERS 

final_pred_output = []

for row in ListStations:

    n_pred_output = copy.copy(config.pred_output)

    for i in range(len(config.pred_output)):  
           
        n_pred_output[i] = "{} {}".format(n_pred_output[i], row).replace('(','').replace(',)','')       
        
    final_pred_output.extend(n_pred_output)  


print ("\n final_pred_output\n ", final_pred_output)

predictions = Model.model_training  (config.full_training, 
                                    X_data_reshaped,
                                    y_data_scaled,
                                    config.epochs,
                                    config.batch_size,
                                    X_shift_reshaped,
                                    config.model_path,
                                    config.model_weights_path,
                                    y_scaler)

predictions_df = pd.DataFrame(predictions, columns = final_pred_output, index = X_shift_data.index)

pd.set_option('display.max_rows', None)
print ("predictions_df", predictions_df)
print ("predictions_df shape", predictions_df.shape)

predictions_df.to_excel(config.retrain_predictions_output_path, index = True)


PlotGenerator.plot_T_DP_graph   (final_pred_output, 
                                predictions_df, 
                                y_test_data, 
                                config.fig_save_path)

PlotGenerator.plot_labels_graph (final_pred_output, 
                                predictions_df, 
                                y_test_data)