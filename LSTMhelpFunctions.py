import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime
from datetime import timedelta

# PREPROCESSING
from sklearn.preprocessing import MinMaxScaler

# SAVE / LOAD MINMAXSCALER
import joblib 

if __name__ == "__train_test_split__":
    train_test_split ()

if __name__ == "__adjust_data__":
    adjust_data ()

if __name__ == "__future_time_series__":
    future_time_series ()

if __name__ == "__shift_data__":
    shift_data ()

if __name__ == "__data_scalling__":
    data_scalling ()

if __name__ == "__data_reshape__":
    data_reshape ()

if __name__ == "__extractTimeInfo__":
    extractTimeInfo ()
       
def train_test_split (train_split, X_data, y_data,):   
    
    # SPLITS DATA INTO TRAINING AND TEST (in this case test data will be shiffted later)
        
    num_data = len(X_data)
    num_train = int(train_split * num_data)
    X_train = X_data[0:num_train]
    X_test = X_data[num_train:]
    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]

    return X_train, X_test, y_train, y_test, num_train

def adjust_data (Data, time_steps):
    
    #ADJUST DATA TO TIME STEPS 

    result, remainder = divmod(len(Data), time_steps)
    Data.drop(Data.tail(remainder).index,inplace=True)
    Data_lenght = len(Data)

    return Data, Data_lenght

def future_time_series (data_intervals, X_inputs):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    endtime = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    X_shift_index = pd.date_range(now, endtime, freq = data_intervals)
    X_shift_index = X_shift_index.floor(data_intervals)
    X_shift_data = pd.DataFrame(columns = X_inputs, index = X_shift_index)

    return X_shift_data
    
def shift_data (y_test):

    # SHIFTS 80% OF TEST DATA INTO FUTURE TIMESTAMPS FOR PREDICTION
    # FUNCTION DEPRECATED 
    # The function future_time_series proved to be a more effective way to generate future time series

    y_data_shift = y_test
    num_shift = len(y_data_shift)
    shift_steps = int(num_shift * 0.2) 
    y_data_shift = y_data_shift.shift(-shift_steps) #allways uncommented
    y_data_shift = y_data_shift.shift(2, freq = 'D')
    y_data_shift = y_data_shift.shift(1, freq = 'H')
    y_data_shift = y_data_shift.shift(50, freq = 'T')

    return y_data_shift, shift_steps

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

def extractTimeInfo (df):

    #FUNCTION TO EXTRACT AS A COLUMN DATE FEATURES (YEAR/MONTH/DAY/HOUR) FROM DATAFRAME INDEX

    df['Date'] = df.index
    df['Date Seq'] = df.index.strftime("%s").astype(int)
    df["Year"] = pd.to_datetime(df["Date"]).dt.strftime("%y").astype(int)
    df["Month"] = pd.to_datetime(df["Date"]).dt.strftime("%m").astype(int)
    df["Day"] = pd.to_datetime(df["Date"]).dt.strftime("%d").astype(int)
    df["Hour"] = pd.to_datetime(df["Date"], format='%H:%M:%S').dt.hour.astype(int)
    del df["Date"]
    return df