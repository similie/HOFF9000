# MODEL CONFIG
import ModelConfig as config

# THE USUAL SUSPECTS
import psycopg2
import pandas as pd
import numpy as np

# CUSTOM FUNCTIONS
import LSTMhelpFunctions 

# Enviroment Variables
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__GetListStation__":
    GetListStation ()

if __name__ == "__MergeStationsData__":
    MergeStationsData()

if __name__ == "__select_query__":
    select_query()

connect_db_info = os.getenv("connect_db_info")

def GetListStation (station, limit):

        conn = psycopg2.connect(connect_db_info)
        GetStations = conn.cursor()

        GetStations.execute("SELECT * FROM public.station WHERE station_type = 1 AND station_state != 979;")
        StationsList = GetStations.fetchall()

        StationsData = {
                'Local Name' : [item[1] for item in StationsList],
                'Station ID' : [item[18] for item in StationsList]
                }

        StationsDF = pd.DataFrame(StationsData, columns = ["Local Name", "Station ID"])

        print ("\n\n",StationsDF)

        # GET STATION AND NEARBY STATIONS (ACCORDING TO LIMIT) GEO LOCATION

        GetStations.execute("SELECT ST_AsText(station.geo) FROM public.station WHERE id = %s;", (station,))
        StationGeo = GetStations.fetchone()
        StationGeo =  ''.join(StationGeo) 
        StationGeo = StationGeo.replace("'", "")
        print (StationGeo)
        GetStations.execute("SELECT id FROM public.station WHERE station_state != 979 ORDER BY ST_Distance(station.geo,'SRID=4326; {}') LIMIT {};".format(StationGeo, limit))
        ListStations = GetStations.fetchall()
        return ListStations 

def MergeStationsData   (ListStations,
                        full_training,  
                        y_inputs, 
                        X_inputs, 
                        data_intervals, 
                        last_dataset_row_path,
                        last_retrain_dataset_row_path):
        
        conn = psycopg2.connect(connect_db_info)
        GetStations = conn.cursor()

        df = pd.DataFrame(columns = {"Date"})

        for row in ListStations:

            if full_training is True:
                dataset_query = select_query (config.station_type, config.full_training)
                GetStations.execute (dataset_query.format (repr(y_inputs)
                                        .replace("'", "")
                                        .replace('[','')
                                        .replace(']','')
                                        .replace('T-DP Variance', 'temperature - dew_point as variance')
                                        , row)
                                        .replace('(','')
                                        .replace(',)',''))

                Data = GetStations.fetchall()
                df = sort_dataset (df, Data, y_inputs, row)
            
            elif full_training is False: 
                # Check last saved table to learn the last index entry
                # Database query will start from that entry
                #last_timestamp_df = pd.read_excel(last_dataset_row_path, index_col='Date')
                #last_timestamp = last_timestamp_df.index.item()
                last_timestamp = "'2021-02-11 16:30:00'" #30min Dare is missing last row
                #last_timestamp = '2021-02-12 15:21:00'
                #last_timestamp = "'2021-02-16 14:00:00'" #30min Dili is missing last row
                print(last_timestamp)

                dataset_query = select_query (config.station_type, config.full_training)
                GetStations.execute(dataset_query.format (repr(y_inputs)
                                                                    .replace("'", "")
                                                                    .replace('[','')
                                                                    .replace(']','')
                                                                    .replace('T-DP Variance', 'temperature - dew_point as variance')
                                                                    , row
                                                                    , last_timestamp)
                                                                    .replace('(','')
                                                                    .replace(',)',''))
                                                                    
                Data = GetStations.fetchall()
                df = sort_dataset (df, Data, y_inputs, row)

            else:
                print("Error: Full Training variable is invalid")
                quit()
                
        pd.set_option('display.max_columns', None)    
        del df['index']
        df.set_index('Date', inplace=True)        
        df = df.resample(data_intervals).median().round(2).dropna(0)
        df = LSTMhelpFunctions.extractTimeInfo(X_inputs, df)
            
        save_xls = df.tail(1)
        save_xls.to_excel(last_retrain_dataset_row_path)

        X = df[X_inputs]
        y = df.sort_index(axis = 0)
        y = y.drop(columns = X_inputs)

        return X, y    

def sort_dataset (df, Data, y_inputs, row):

    DataSet = {
               'Date': [item[0] for item in Data],
            }

    for i in range(len(y_inputs)):
        y_inputs[i] = "{} {}".format(y_inputs[i], row).replace('(','').replace(',)','')
        feature = {
            y_inputs[i]: [item[i+1] for item in Data],
        }
        DataSet.update(feature)
                
    pd.set_option('display.max_columns', None)
    df_db = pd.DataFrame(DataSet, columns = {"Date"})
                
    for x in range(len(y_inputs)):
        merge_column = pd.DataFrame(DataSet, columns = {"Date", y_inputs[x]})  
        df_db = pd.merge_ordered(df_db, merge_column, how="outer") 
    
    df_db.reset_index(inplace=True)
    df = pd.merge_ordered(df, df_db, how="outer")   
    print ("\nLOADING...\n")
    
    return df

def select_query (station_type, full_training):
# Select PSQL Queries

    if station_type == "water" and full_training == True:
        dataset_query = """SELECT date AT TIME ZONE 'Asia/Dili', {}
                                    FROM lawful_curtain_6169.wl WHERE station = {}
                                    AND percent_full > 0;
                                    """

    elif station_type == "water" and full_training == False:
        dataset_query = """SELECT date AT TIME ZONE 'Asia/Dili', {}
                                    FROM lawful_curtain_6169.wl WHERE station = {}
                                    AND date AT TIME ZONE 'Asia/Dili' > {}
                                    AND percent_full > 0;
                                    """

    elif station_type == "weather" and full_training == True:
        dataset_query = """SELECT date AT TIME ZONE 'Asia/Dili', {}
                                    FROM assets.all_weather WHERE station = {}
                                        AND pressure > 800
                                        AND precipitation < 6
                                        AND humidity < 101
                                        AND temperature > 10
                                        AND temperature < 100
                                        AND dew_point > 16
                                        AND wind_speed > 0
                                        ORDER BY date ASC;
                                        """

    elif station_type == "weather" and full_training == False:
        dataset_query = """SELECT date AT TIME ZONE 'Asia/Dili', {}
                                    FROM assets.all_weather WHERE station = {}
                                        AND date AT TIME ZONE 'Asia/Dili' > {}
                                        AND pressure > 800
                                        AND precipitation < 6
                                        AND humidity < 101
                                        AND temperature > 10
                                        AND temperature < 100
                                        AND dew_point > 16
                                        AND wind_speed > 0
                                        ORDER BY date ASC;
                                        """

    else: 
        print ("ERROR: station_type or full_training are incorrect or don't match")
    
    return dataset_query