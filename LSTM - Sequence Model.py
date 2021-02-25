###################################################
#                   MAIN FILE                     #
###################################################
###################################################
#            HOFF 9000 RUNS FROM HERE             #
###################################################

# MODEL CONFIG
import ModelConfig as config

# THE USUAL SUSPECTS
import pandas as pd
import numpy as np

# SSH TUNNEL
from sshtunnel import SSHTunnelForwarder

# CUSTOM FUNCTIONS
import DBQueryFunctions  
import LSTMhelpFunctions 
import PlotGenerator
import Model

# SSH TUNNEL TO BARE METAL SERVER
with SSHTunnelForwarder(  
    config.IP_Address,
    ssh_username = config.username,
    ssh_pkey = config.PEM_Location,
    remote_bind_address = (config.bind_address, config.port),
    local_bind_address = (config.bind_address, config.port)):
    
    print ('SSH CONNECTION STARTED')

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

print ("X_data\n", X_data)
print ("X_data shape\n", X_data.shape)
print ("y_data\n", y_data)
print ("y_data shape\n", y_data.shape)

# Train-Test Split
num_data = len(X_data) - config.shift_steps
X_test_data = X_data[num_data:] 
y_test_data = y_data[num_data:] 

print ("y_test_data", y_test_data)
print("y_test_data shape", y_test_data.shape)

# FUTURE TIME SERIES SEQUENCE
X_shift_data = LSTMhelpFunctions.future_time_series (config.hours,
                                                    config.data_intervals, 
                                                    config.X_inputs)

X_shift_data = LSTMhelpFunctions.extractTimeInfo(config.X_inputs, X_shift_data)

print ("X_shift_data", X_shift_data)
print ("X_shift_data shape", X_shift_data.shape)

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

print ("X_shift_reshaped", X_shift_reshaped)

predictions = Model.model_training  (config.full_training, 
                                    X_data.shape[0], 
                                    X_data.shape[1],
                                    y_data.shape[1],
                                    X_data_reshaped,
                                    y_data_scaled,
                                    config.epochs,
                                    config.batch_size,
                                    X_shift_reshaped,
                                    config.model_path,
                                    config.model_weights_path,
                                    y_scaler)

pred_output = config.pred_output

for row in ListStations:
    for i in range(len(pred_output)):
        pred_output[i] = "{} {}".format(pred_output[i], row).replace('(','').replace(',)','')

predictions_df = pd.DataFrame(predictions, columns = pred_output, index = X_shift_data.index)

print ("predictions_df", predictions_df)
print ("predictions_df shape", predictions_df.shape)

predictions_df.to_excel(config.retrain_predictions_output_path, index = True)

PlotGenerator.plot_T_DP_graph   (pred_output, 
                                predictions_df, 
                                config.y_inputs,
                                y_test_data, 
                                config.fig_save_path)

PlotGenerator.plot_labels_graph (pred_output, 
                                predictions_df, 
                                config.y_inputs, 
                                y_test_data)