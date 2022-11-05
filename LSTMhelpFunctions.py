import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime
from datetime import timedelta

# PREPROCESSING
from sklearn.preprocessing import MinMaxScaler

# SAVE / LOAD MINMAXSCALER
import joblib 


if __name__ == "__future_time_series__":
    future_time_series ()

if __name__ == "__data_scalling__":
    data_scalling ()

if __name__ == "__data_reshape__":
    data_reshape ()

if __name__ == "__extractTimeInfo__":
    extractTimeInfo ()
       

def future_time_series (hours, data_intervals, X_inputs):

    #now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now = "'2022-09-20 04:00:00'" #just for testing
    #endtime = (datetime.now() + timedelta(hours = hours)).strftime("%Y-%m-%d %H:%M:%S")
    endtime = "'2022-09-21 04:00:00'" #just for testing
    X_shift_index = pd.date_range(now, endtime, freq = data_intervals)
    X_shift_index = X_shift_index.floor(data_intervals)
    X_shift_data = pd.DataFrame(columns = X_inputs, index = X_shift_index)

    return X_shift_data
    
def data_scalling   (full_training, 
                    X_data, 
                    X_shift_data, 
                    y_data, 
                    X_scaler_path, 
                    y_scaler_path):
    
    print("full_training",full_training)

    if full_training is True:
        print("test True")
        X_scaler = MinMaxScaler(feature_range = (0, 1))
        y_scaler = MinMaxScaler(feature_range = (0, 1))

        X_data_scaled = X_scaler.fit_transform(X_data)
        X_shift_scaled = X_scaler.transform(X_shift_data)
        
        y_data = np.array(y_data)
        y_data_scaled = np.reshape(y_data, (y_data.shape[0],  y_data.shape[1]))
        y_data_scaled = y_scaler.fit_transform(y_data_scaled)

        #Save Scaler
        joblib.dump(X_scaler, X_scaler_path) 
        joblib.dump(y_scaler, y_scaler_path) 

    elif full_training is False:
        print("test False")
        #Load Scaler
        X_scaler = joblib.load(X_scaler_path) 
        y_scaler = joblib.load(y_scaler_path)

        X_data_scaled = X_scaler.fit_transform(X_data)
        X_shift_scaled = X_scaler.transform(X_shift_data)
        
        y_data = np.array(y_data)
        y_data_scaled = np.reshape(y_data, (y_data.shape[0],  y_data.shape[1]))
        y_data_scaled = y_scaler.fit_transform(y_data_scaled)
    
    else:
        print("Error: Full Training variable is invalid")
        quit()
   
    return X_data_scaled, X_shift_scaled, y_data_scaled, y_scaler

def data_reshape (X_train_scaled, X_test_scaled, time_steps, X_train_length, X_test_length):

    # RESHAPES INPUT DATA TO 3 DIMENSIONS
    # Shape the input data into a dimension that can be used by the model (Number of Rows, Time-Steps, Number of Features)
    # FEATURE DEPRECATED - keepin it for a potential time-steps scenario

    X_train_scaled, X_test_scaled = np.array(X_train_scaled), np.array(X_test_scaled)
    X_train_reshaped = np.reshape(X_train_scaled, ((X_train_length // time_steps), time_steps, X_train_scaled.shape[1]))
    X_test_reshaped = np.reshape(X_test_scaled, ((X_test_length // time_steps), time_steps, X_test_scaled.shape[1]))

    return X_train_reshaped, X_test_reshaped

def extractTimeInfo (X_inputs, df):

    #FUNCTION TO EXTRACT DATE FEATURES (YEAR/MONTH/DAY/HOUR) FROM DATAFRAME INDEX

    df['Date'] = df.index
    for column in X_inputs:
        if column == 'Date Seq':
            df['Date Seq'] = df.index.strftime("%s").astype(int)
        elif column == 'Year':  
            df["Year"] = pd.to_datetime(df["Date"]).dt.strftime("%y").astype(int)
        elif column == 'Month':
            df["Month"] = pd.to_datetime(df["Date"]).dt.strftime("%m").astype(int)
        elif column == 'Day':
            df["Day"] = pd.to_datetime(df["Date"]).dt.strftime("%d").astype(int)
        elif column == 'Hour':
            df["Hour"] = pd.to_datetime(df["Date"], format='%H:%M:%S').dt.hour.astype(int)
        else: 
            print ("Error: You have choosen an inexistent DateTime/Sequence variable")
            quit()

    del df["Date"]
    return df