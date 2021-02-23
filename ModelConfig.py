###################################################
#        MAIN TRAINNING MODEL CONFIGURATION       #
###################################################
###################################################
# MODEL 30 MIN INTERVALS / BATCH 1 / FORECAST 24h #
###################################################
import os
from dotenv import load_dotenv

load_dotenv()

# SSH TUNNEL CONFIG
IP_Address = os.getenv("IP_Address")
username = os.getenv("username")
PEM_Location = os.getenv("PEM_Location")
bind_address = os.getenv("bind_address")
port = int(os.getenv("port"))

# MODEL VARIABLES
full_training = False   # SET TO True IF IT IS THE FIRST TIME RUNNING
                        # SET full_training TO True IN ORDER TO RUN A COMPLETE RUN ON ALL DATA
                        # SET full_training TO False IN ORDER TO RETRAIN MODEL FROM LAST TIME STAMP
 
station = 27            # SELECT STATION ID FOR ANALYSIS Ex: Dare = 4; Similie = 27 (check database for other stations)
limit = 1               # SELECT THE NUMBER OF NEARBY STATIONS TO MERGE IN THE ALGORITHM
data_intervals = '30T'  # SELECT THE DATA INTERVALS FOR THE DATA 10T/30T/1H/1D etc (T = minutes)
shift_steps = 48        # data_intervals * shift_steps = X DAYS TO LOOK AHEAD
batch_size = 1          # NUMBER OF SAMPLES PROPAGATED THROUGH THE NETWORK
epochs = 100            # NUMEBR OF TIMES THE MODEL CYCLES THROUGH THE FULL TRAINING DATASET

#DEFINE X AND Y PARAMETERS 

X_inputs = ["Date Seq", "Hour", "Day", "Month", "Year"] # FEATURES USED FOR TIME SERIES PREDICTIONS

y_inputs = ["Temperature", "Dew Point", "T-DP Variance", "Humidity", "Pressure", "Wind Speed", "Wind Direction"] # FEATURES TO BE PREDICTED

pred_output = ["Pred Temperature", "Pred Dew Point", "Pred T-DP Variance", "Pred Humidity", "Pred Pressure", "Pred Wind Speed", "Pred Wind Direction"] # PREDICTED OUTPUTS 


# FULL TRIANAIN MODEL FILE PATHS

last_dataset_row_path = '{}/last_dataset_row.xls'.format(os.getenv("last_dataset_row_path"))
X_scaler_path = '{}/Model_Scaler_X.gz'.format(os.getenv("X_scaler_path"))                               #USED IN TRAIN AND RETRAIN
y_scaler_path = '{}/Model_Scaler_y.gz'.format(os.getenv("X_scaler_path"))                               #USED IN TRAIN AND RETRAIN
model_path = '{}/LSTM_Model_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("model_path"))     #USED IN TRAIN AND RETRAIN
model_weights_path = '{}/LSTM_Model_WEIGHTS_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("model_weights_path"))
predictions_output_path = '{}/Sequence_24hForecast_Output.xls'.format(os.getenv("predictions_output_path"))
fig_save_path = '{}/Pred_T_and_DP.png'.format(os.getenv("predictions_output_path"))

# RETRAIN MODEL SPECIFIC FILE PATHS

last_retrain_dataset_row_path = '{}/last_retrain_dataset_row.xls'.format(os.getenv("last_retrain_dataset_row_path"))
retrain_predictions_output_path  = '{}/Retrain_Output.xls'.format(os.getenv("retrain_predictions_output_path"))
retrain_figure1_save = '{}/Retrain_Pred_T_and_DP.png'.format(os.getenv("retrain_figure1_save"))