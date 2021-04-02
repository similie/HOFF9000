# HOFF9000

[<img src="https://user-images.githubusercontent.com/29231033/113271826-e3c70780-9315-11eb-8243-8f7f5b35dbed.png" width="100px" />](https://user-images.githubusercontent.com/29231033/113271826-e3c70780-9315-11eb-8243-8f7f5b35dbed.png)
&nbsp;&nbsp;&nbsp;&nbsp;
[<img src="https://user-images.githubusercontent.com/5084787/113365120-3c3aeb00-9390-11eb-8d9f-27cc1fd31122.png" height="100px" />](https://user-images.githubusercontent.com/5084787/113365120-3c3aeb00-9390-11eb-8d9f-27cc1fd31122.png)


**Project Overview:**
"Did anyone tell you... not to hassle the Hoff 9000?"

This repository is a collaboration project between [Similie](https://similie.org) and [Mercy Corps](https://www.mercycorps.org/). We are building a localized early warning solution for data-poor economies and started the HOFF9000 project with the goal of leveraging machine learning algorithms geared toward predicting various indicators relevant in the detection of flash floods and other natural disaster occurrences. In the event of natural disaster, we seek to build affordable solutions where individuals living in underserved economies can have advanced warnings required to move their resources and their families to safety. HOFF9000 is a machine-learning component that offers predictive outputs of local environmental data aimed at improving our time before event metrics. Why HOFF9000? Well, it comes from one of our favorite movies, and we needed a name.

In its current iteration, we use data from a rang of low-cost IoT sensors gathering precipitation, river water levels, soil moisture, and other atmospheric parameters to train forecasted precipitation outputs. Our next phases will attempt to use these prediction models to adjust thresholds for what constitutes a potential early warning event. For example, when precipitation at station A records X and the water level at station B records Y, what is the probability of an event for area C? X and Y are optimizations that HOFF9000 will attempt to solve.
 
This initial attempt was trialed in Timor-Leste where the lack of adequate weather catchment data, along with  deforestation, make for extreme and highly unpredictable impacts throught extreme weather patterns. 

This initial attempt was trialed in Timor-Leste where the lack of adequate weather, water catchment, and other environmental data is scarce. Compounded by increasingly unpredictable climate change-based weather patterns and human-induced deforestation, flash flooding events have become a devastating force in an already vulnerable and resource poor economy. For example, on March 13th of 2020, Timor-Leste experienced an extreme weather event which cost millions of dollars for the Dili urban area.

![image](https://user-images.githubusercontent.com/29231033/113268610-867d8700-9312-11eb-999c-3f0d41a38868.png)

The event  data was recorded by weather stations owned  by GoTL and Similie.

![image](https://user-images.githubusercontent.com/29231033/113271869-f2adba00-9315-11eb-881b-6307b4ba3d9a.png)

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

**Notes and Contributions**
Currently, the solutions provided in this repository should be considered experimental and for development purposes only. Because this is a truly worthwhile cause and can have an impact on the livelihoods of millions living in underserved economies, we actively encourage contributions to the HOFF9000 project, and believe this to be a potential showcase into the power of opensource communities. Therefore, pull requests will be actively reviewed, tested, and accepted by the repository maintainers. 
