# Time Series Analysis using ARIMA, LSTM, and Prophet

This repository contains a Python script that performs time series analysis on a sample dataset using three different models - ARIMA, LSTM, and Prophet.

## Requirements
- Python 3.7 or higher
- pandas
- numpy
- matplotlib
- plotly
- statsmodels
- tensorflow
- scikit-learn
- prophet

## Dataset
The dataset used for this analysis is a sample dataset named `sample_1.csv`. It contains two columns - `point_timestamp` and `point_value`. The `point_timestamp` column contains timestamps and the `point_value` column contains numerical values.

## Usage
1. Clone this repository or download the `time_series_analysis.py` file.
2. Open a terminal and navigate to the directory where `time_series_analysis.py` is located.
3. Run the script by typing `python time_series_analysis.py` in the terminal.

## Description of the Models
- ARIMA: ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting method that uses past values of the series to predict future values. It is a popular statistical model for time series analysis.
- LSTM: LSTM (Long Short-Term Memory) is a type of neural network that is used for sequence prediction problems. It is well suited for time series forecasting as it can handle the long-term dependencies and can remember the past information.
- Prophet: Prophet is a forecasting library developed by Facebook that is used for time series analysis. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

## Results
The script generates line plots of the original data and the predicted values for each model. It also prints the summary of the ARIMA model and the evaluation metrics for each model, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Conclusion
This script provides an example of how to perform time series analysis using different models and evaluate their performance. The results can be used to determine which model is best suited for a particular dataset and to make accurate predictions for future values.
