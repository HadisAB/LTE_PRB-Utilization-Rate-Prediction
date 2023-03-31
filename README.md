# LTE_PRB-Utilization-Rate-Prediction
In this project we are supposed to predict the PRB utilization rate of LTE sites in a mobile telecommunication network.

## Goal
As the sites usage is changing in special events, it is a good idea to predict their behaviour, find the congested sites and locate cow-sites in their area or plan new sites. So in this project we applied a method to predict the 'PRB_Utilization_Rate' weekly KPI per site and find future congestions in the network.

## Method
- Export data from DB or relevant tools.
- Apply preprocessing methods to clean the data and make it ready for prediction.
- As 'PRB_Utilization_Rate' is a time-series KPI, first we need to check the stationary status of data.
- Make the KPI stationary to be prepared for 'ARIMA' model.
- Apply Train-Test-Split method.
- Apply Arima Model to the dataframe.
- Predict new data and check the MSE (Total MSE: 5.5) for validation.


<img src=https://github.com/HadisAB/LTE_PRB-Utilization-Rate-Prediction/blob/main/PredictionPlot.jpg/>
