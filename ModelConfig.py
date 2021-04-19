###################################################
#        MAIN TRAINNING MODEL CONFIGURATION       #
###################################################
###################################################
# MODEL 30 MIN INTERVALS / BATCH 1 / FORECAST 24h #
###################################################
import os
from dotenv import load_dotenv

load_dotenv()

###########################
#### SSH TUNNEL CONFIG ####
###########################

IP_Address = os.getenv("IP_Address")
username = os.getenv("username")
PEM_Location = os.getenv("PEM_Location")
bind_address = os.getenv("bind_address")
port = int(os.getenv("port"))

#########################
#### MODEL VARIABLES ####
#########################

full_training = False   # SET full_training = True IF IT IS THE FIRST TIME RUNNING.
                        # SET full_training = True IN ORDER TO RUN A COMPLETE RUN ON ALL DATA
                        # SET full_training = False IN ORDER TO RETRAIN MODEL FROM LAST TIME STAMP
 
station = 4             # SELECT STATION ID FOR ANALYSIS Ex: Dare = 4; Similie = 27 (check database for other stations)
limit = 1               # SELECT THE NUMBER OF NEARBY STATIONS TO MERGE IN THE ALGORITHM
hours = 24              # NUMBER OF HOURS FOR FORECAST 
data_intervals = '30T'  # SELECT THE DATA INTERVALS FOR THE DATA 10T/30T/1H/1D etc (T = minutes)
shift_steps = 48      # data_intervals * shift_steps = X DAYS TO LOOK AHEAD
batch_size = 1          # NUMBER OF SAMPLES PROPAGATED THROUGH THE NETWORK
epochs = 100            # NUMEBR OF TIMES THE MODEL CYCLES THROUGH THE FULL TRAINING DATASET

###################################
#### DEFINE X AND Y PARAMETERS ####
###################################

# X_inputs ARE SPECIFIC FEATURES USED FOR TIME SERIES PREDICTIONS
# THEY NEED TO BE TIME RELATED, SEQUENTIAL AND PREDICTABLE
# CURRENT CODE ONLY SUPPORTS 5 OPTIONS: "Date Seq", "Hour", "Day", "Month", "Year".

X_inputs = ["Date Seq", "Hour", "Day", "Month", "Year"] 
#X_inputs = ["Date Seq", "Hour"]

# CURRENTLY MODEL WORKS FOR WEATHER ("weather") OR WATER TANKS ("water"). BE SURE TO SELECT THE CORRECT  TYPE.
station_type = "weather"

# y_inputs ARE THE FEATURES IN THE SEQUENTIAL MODEL THAT YOU WANT TO RELATE TO A TIME SEQUENCE (X_inputs) 
# WEATHER FEATURES AVAILABLE  "temperature", "dew_point", "T-DP Variance", "humidity", "pressure", "wind_speed", "wind_direction"]

y_inputs = ["temperature", "dew_point", "T-DP Variance", "humidity", "pressure", "wind_speed", "wind_direction"] 

# WATER TANK FEATURES AVAILABLE 

#y_inputs = ["percent_full", "tank_health", "liters", "water_level"] 

# pred_output FEATURES THAT WILL BE PREDICTED

pred_output = ["Pred Temperature", "Pred Dew Point", "Pred T-DP Variance", "Pred Humidity", "Pred Pressure", "Pred Wind Speed", "Pred Wind Direction"] # PREDICTED OUTPUTS FOR WEATHER
#pred_output = ["Pred percent_full", "Pred tank_health", "Pred liters", "Pred water_level"]

############################
# FULL TRAINING FILE PATHS #
############################

last_dataset_row_path = '{}/last_dataset_row.xls'.format(os.getenv("last_dataset_row_path"))
X_scaler_path = '{}/Model_Scaler_X.gz'.format(os.getenv("X_scaler_path"))                               
y_scaler_path = '{}/Model_Scaler_y.gz'.format(os.getenv("X_scaler_path"))                               
model_path = '{}/LSTM_Model_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("model_path"))     
model_weights_path = '{}/LSTM_Model_WEIGHTS_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("model_weights_path"))
predictions_output_path = '{}/Sequence_24hForecast_Output.xls'.format(os.getenv("predictions_output_path"))
fig_save_path = '{}/Pred_T_and_DP.png'.format(os.getenv("predictions_output_path"))

# RETRAINING FILE PATHS

last_retrain_dataset_row_path = '{}/last_retrain_dataset_row.xls'.format(os.getenv("last_retrain_dataset_row_path"))
retrain_predictions_output_path  = '{}/Retrain_Output.xls'.format(os.getenv("retrain_predictions_output_path"))
retrain_figure1_save = '{}/Retrain_Pred_T_and_DP.png'.format(os.getenv("retrain_figure1_save"))