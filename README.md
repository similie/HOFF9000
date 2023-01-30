# Tabula Rasa

[<img src="https://user-images.githubusercontent.com/29231033/113271826-e3c70780-9315-11eb-8243-8f7f5b35dbed.png" width="100px" />](https://user-images.githubusercontent.com/29231033/113271826-e3c70780-9315-11eb-8243-8f7f5b35dbed.png)
&nbsp;&nbsp;&nbsp;&nbsp;

**Project Overview:**
"Tabula rasa is the theory that individuals are born without built-in mental content, and therefore all knowledge comes from experience or perception."

This repository is a collaboration project between [Similie](https://similie.org). We are building a localized early warning solution for data-poor economies and started the Tabula Rasa project with the goal of leveraging machine learning algorithms geared toward predicting various indicators relevant in the detection of flash floods and other natural disaster occurrences. In the event of natural disaster, we seek to build affordable solutions where individuals living in underserved economies can have advanced warnings required to move their resources and their families to safety. Tabula Rasa is a machine-learning component that offers predictive outputs of local environmental data aimed at improving our time before event metrics. 

In its current iteration, we use data from a rang of low-cost IoT sensors gathering precipitation, river water levels, soil moisture, and other atmospheric parameters to train forecasted precipitation outputs. Our next phases will attempt to use these prediction models to adjust thresholds for what constitutes a potential early warning event. For example, when precipitation at station A records X and the water level at station B records Y, what is the probability of an event for area C? X and Y are optimizations that Tabula Rasa will attempt to solve.
 
This initial attempt was trialed in Timor-Leste where the lack of adequate weather, water catchment, and other environmental data is scarce. Compounded by increasingly unpredictable climate change-based weather patterns and human-induced deforestation, flash flooding events have become a devastating force in an already vulnerable and resource poor economy. For example, on March 13th of 2020, Timor-Leste experienced an extreme weather event which cost millions of dollars for the Dili urban area.

![image](https://user-images.githubusercontent.com/29231033/113268610-867d8700-9312-11eb-999c-3f0d41a38868.png)

The event  data was recorded by weather stations owned  by GoTL and Similie.

![image](https://user-images.githubusercontent.com/29231033/113271869-f2adba00-9315-11eb-881b-6307b4ba3d9a.png)

**How to run it:**

1. *Install Python:*
> brew install python

Note: Brew should install Pip 

2. *Create an venv:*
> python3 -m venv /path/to/new/virtual/environment

3. *Activate venv:*
> source <venv>/bin/activate

4. *Install Requirements (libraries):*
> python -m pip install requirements.txt

5. *Run script LSTM_Sequence_Model.py:*
> python -m LSTM_Sequence_Model

6. *Run the script with logs (reccomendded):*
> python -m LSTM_Sequence_Model > /path/to/logs.txt

**Model Overview:**
- Long Short-Term Memory (LSTM) - Recurrent Neural Network (Deep Learning). 
- LSTM has been widely used for weather forecasting algorithms, energy and water consumption and stock market prediction models. 
- Base Features: Temperature, Dew Point, Temperature and Dew Point Variance,  Humidity, Pressure, Wind Speed and Wind Direction.

**Outcomes:**
- 24 hour forecast values for Temperature and Dew Point. 
- The convergence of Temperature and Dew Point predictions indicates a strong probability of precipitation. 
- Output accuracy is currently estimated at 79.29%. 

![image](https://user-images.githubusercontent.com/29231033/113272586-bcbd0580-9316-11eb-9640-a15372536d04.png)

**Improve current models:**
- Move to production can increase the models self-learning rate without human intervention.
- Learn and integrate better machine learning evaluation techniques. 
- Integrate Global Weather Forecast Models with our ML Models. 

**Impact on Flood EWS:**
- Incorporate future data from Dili EWS project, such as soil moisture, river level data and local weather conditions into existing models.
- Assess results alongside standard approaches to understand if AI presents a benefit.
- Test ML Models in new catchments.
- Develop a communications approach to improve understanding within Government.
- Improve visualization of AI results in One platform for EWS.

**Explore AI in new concepts:**
- Predict impact of time without rain on rainfall/runoff curve.  
- Understand water availability within a catchment.
- Forecast Fire Index values.
- Use satellite timeseries images to better understand climate trends.

**How it works**
- Tabula Rasa is built to integrate with Similie platform ONE. ONE collects different types of data including Weather Data.
- It was made so it could continuously learn from ONE database without the need of human intervention.

**Python Libraries used**
- joblib==0.17.0
- numpy==1.18.5
- psycopg2==2.8.6
- scikit-learn==0.23.2
- scipy==1.5.4
- threadpoolctl==2.1.0
- matplotlib==3.3.2
- pandas==1.1.3
- pandas-compat==0.1.1
- sshtunnel==0.4.0
- Keras==2.4.3
- tensorflow==2.3.1
- xlwt==1.3.0
- python-dotenv

**Python Files Architecture**

LSTM - Sequence Model.py:
- Main file and backbone of the script
- Start the script from here
- It is used to calls other files and functions that will colect, shape, transform and inject the data into the model.

ModelConfig.py:
- Configuration file
- Ideally its the only file that a user needs to change
- You can define wether you are making a FULL trainning of a new model or Retraining an older model.
- Contains the paths needed to save: the scaler, model weights, generated predictions etc
- Here you can define the most important parameters to run the Sequential Model. Which Station, how many hours to forecast, size of data intervals, batchsize, number of epochs etc
- It is possible select the time series features (X Inputs) used for the LSTM Model such as:
  - "Date Seq",
  - "Hour", 
  - "Day", 
  - "Month", 
  - "Year"

- Currently you can select only from these 5 features.
- More time series features can be added to the code such as trimesters, seasons, fortnights or other time trends. However this will need some code manipulation.

- You can select the station type. Currently the station types available in ONE are Weather (for weather stations) and Water (for water tanks).

- It is possible select the features which you want to relate to the time series (Y Inputs). Currently available features are:

- **For Weather Stations:**
  - "temperature",
  - "dew_point",
  - "T-DP Variance" (temperature and dewpoint variance),
  - "humidity", 
  - "pressure", 
  - "wind_speed", 
  - "wind_direction".
  
- **For Water Tanks:**
  - "percent_full", 
  - "tank_health", 
  - "liters", 
  - "water_level"

- **For River Level:**
  - "water_level"
  - Note: Water Tanks sensors are similar ultrasonic sensors used in River Level therefor the same features may apply.
  However since river conditions are different from a tank we reccomend to use only the "water_level" feature.
  
- There are more features available but these are the ones successfully tested.
- You can select only one or more features related with the station type choosen.

- You need also to define the prediction output which should be closely related to Y Inputs:
  Possible example for the Weather Station:
  - "Pred Temperature", 
  - "Pred Dew Point", 
  - "Pred T-DP Variance", 
  - "Pred Humidity", 
  - "Pred Pressure", 
  - "Pred Wind Speed", 
  - "Pred Wind Direction".
 
DBQueryFunctions.py:
- Queries ONE Platform DB according to the options selected in ModelConfig.py
- This is the most important file since it is where the data is cleaned, sorted and merged for further usage.

LSTMHelpFunctions.py:
- It is an assortment of different useful functions that help the previous files work
- Examples:
  - Generate time series (future_time_series)
  - Scalling data between 0 and 1 (data_scalling)
  - Reshape Data (data_reshape)
  - Etc
- Note: Some of the functions present in this file may be deprecated.

Model.py:
- Here is where the Neural Network is defined by using the KERAS Library.
- The model present in this file can easily be changed agnostically from the rest of the script
- The Neural Network is currently defined as follows:
  - model = Sequential() #MODEL TYPE
  - model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_data_shape_0, X_data_shape_1))) #First LSTM layer   
  - model.add(LSTM(units = 50, return_sequences = True)) # Second LSTM layer 
  - model.add(LSTM(units = 50, return_sequences = True)) # Third LSTM layer
  - model.add(LSTM(units = 50)) # Fourth LSTM layer  
  - model.add(Dense(units = y_data_shape_1, activation = 'relu'))  # Output layer
  - model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy']) # Compiling the RNN
  - model.fit(X_data_reshaped, y_data_scaled, epochs = epochs, batch_size = batch_size) # Fitting the RNN to the Training set

- So far this has been the Neural Network that has provided the best results. However feel free to try other Neural Networks structures!

PlotGenerator.py:
- This file's purpose is to generate graphs for easier visualization of the predictions.
- It is not required in Production Environments

**Currently Missing from Repository**
- Unfortunatelly this repository does not yet contain the methods to execute proper Evaluation of the Deep Learning Model. We are hoping to add it soon!

**Notes and Contributions**
Currently, the solutions provided in this repository should be considered experimental and for development purposes only. Because this is a truly worthwhile cause and can have an impact on the livelihoods of millions living in underserved economies, we actively encourage contributions to the Tabula Rasa project, and believe this to be a potential showcase into the power of opensource communities. Therefore, pull requests will be actively reviewed, tested, and accepted by the repository maintainers. 
