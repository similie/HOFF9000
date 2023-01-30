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

full_training = True    # SET full_training = True IF IT IS THE FIRST TIME RUNNING.
                        # SET full_training = True IN ORDER TO RUN A COMPLETE RUN ON ALL DATA
                        # SET full_training = False IN ORDER TO RETRAIN MODEL FROM LAST TIME STAMP
 
station = 27            # SELECT STATION ID FOR ANALYSIS Ex: Dare = 4; Similie HQ = 27 (check database for other stations)
limit = 2               # SELECT THE NUMBER OF NEARBY STATIONS TO MERGE IN THE ALGORITHM
hours = 24              # NUMBER OF HOURS FOR FORECAST - Needs testing
data_intervals = '10T'  # SELECT THE INTERVALS FOR THE DATA 10T/30T/1H/1D etc (T = minutes)
batch_size = 1          # NUMBER OF SAMPLES PROPAGATED THROUGH THE NETWORK. Batch Size = 1, For Online Training
epochs = 10             # NUMEBR OF TIMES THE MODEL CYCLES THROUGH THE FULL TRAINING DATASET

###################################
#### DEFINE X AND Y PARAMETERS ####
###################################

# X_inputs 
# TIME SERIES FEATURES

X_inputs = ["Date Seq", 
                "Hour", 
                "Day", 
                "Month", 
                "Year",
                "Quarter"] 


# CURRENTLY MODEL WORKS FOR WEATHER ("weather") OR WATER TANKS ("water"). 
station_type = "weather"

# y_inputs 
# TO BE PREDICTED
# WEATHER FEATURES 

y_inputs = ["temperature", 
            "dew_point", 
            "T-DP Variance", 
            "humidity", 
            "pressure", 
            "wind_speed", 
            "wind_direction", 
            "solar"] 

# WATER TANK FEATURES  

#y_inputs = ["percent_full", "tank_health", "liters", "water_level"] 

# PREDICTED FEATURES

# OUTPUT WEATHER 
pred_output = ["Pred Temperature", 
                "Pred Dew Point", 
                "Pred T-DP Variance", 
                "Pred Humidity", 
                "Pred Pressure", 
                "Pred Wind Speed", 
                "Pred Wind Direction", 
                "Pred Solar"] 

# OUTPUT WATER TANKS 

#pred_output = ["Pred percent_full", "Pred tank_health", "Pred liters", "Pred water_level"]
#pred_output = ["Pred water_level"]

############################
# FULL TRAINING FILE PATHS #
############################

model_output_folder = "model_output_folder" 
model_retrain_output_folder = "model_retrain_output_folder"

Keras_Tuner_path = '{}/Keras_Tuner_Optimizer'.format(os.getenv("folder_path")) #OPTIMIZER 

last_dataset_row_path = '{}/{}/last_dataset_row.xls'.format(os.getenv("folder_path"), "model_output_folder")     

X_scaler_path = '{}/{}/Model_Scaler_X.gz'.format(os.getenv("folder_path"), "model_output_folder")                                 
y_scaler_path = '{}/{}/Model_Scaler_y.gz'.format(os.getenv("folder_path"), "model_output_folder")                             
model_path = '{}/{}/LSTM_Model_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("folder_path"), "model_output_folder")    
model_weights_path = '{}/{}/LSTM_Model_WEIGHTS_Forecast_24h_Interval-30min_batch-1.h5'.format(os.getenv("folder_path"), "model_output_folder") 
predictions_output_path = '{}/{}/Sequence_24hForecast_Output.xls'.format(os.getenv("folder_path"), "model_output_folder")  
fig_save_path = '{}/{}/Pred_T_and_DP.png'.format(os.getenv("folder_path"), "model_output_folder") 

#########################
# RETRAINING FILE PATHS #
#########################

last_retrain_dataset_row_path = '{}/{}/last_retrain_dataset_row.xlsx'.format(os.getenv("folder_path"), "model_retrain_output_folder")
retrain_predictions_output_path  = '{}/{}/Retrain_Output.xlsx'.format(os.getenv("folder_path"), "model_retrain_output_folder")
retrain_figure1_save = '{}/{}/Retrain_Pred_T_and_DP.png'.format(os.getenv("folder_path"), "model_retrain_output_folder") 