
import numpy as np   # Library for n-dimensional arrays
import pandas as pd  # Library for dataframes (structured data)

# Helper imports
import os 
import warnings
import pandas_datareader as web
import datetime as dt

# ML/DL imports
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot


# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#from plotly.offline import init_notebook_mode, iplot

# %matplotlib inline
#warnings.filterwarnings("ignore")
#init_notebook_mode(connected=True)

# Set seeds to make the experiment more reproducible.
from numpy.random import seed
seed(1)

df = pd.read_csv("sample_1.csv")

df

df.columns

df = df.iloc[: , 1:]

df.info()

df.describe().transpose()

df.isna().sum()

df

from statsmodels.tsa.arima.model import ARIMA
# Convert the 'point_timestamp' column to a datetime object
#df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])

# Set the 'point_timestamp' column as the dataframe index
df.set_index('point_timestamp', inplace=True)

# Create a time series plot
df.plot(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Point Value')
plt.show()

# Fit an ARIMA model to the data
model = ARIMA(df['point_value'], order=(1, 1, 1))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Make predictions for the next 10 time steps
forecast = model_fit.forecast(steps=10)

# Print the forecasted values
print(forecast)

df

df = pd.read_csv("sample_1.csv")
df = df.iloc[: , 1:]

import plotly.express as px

fig = px.line(df, x='point_timestamp', y='point_value')
fig.update_layout(title='Time Series Data', xaxis_title='Timestamp', yaxis_title='Value')
fig.show()

fig = px.scatter(df, x='point_timestamp', y='point_value', color='point_value')
fig.update_layout(title='Scatter Plot of Time Series Data', xaxis_title='Timestamp', yaxis_title='Value')
fig.show()

# Preprocess the data
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df = df.set_index('point_timestamp')

# Step 3: Split the data into train and test sets

train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Step 4: Build different time series models

# ARIMA model
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(train_data['point_value'], order=(2,1,2))
arima_fit = arima_model.fit()
arima_pred = test_data.copy()
arima_pred['arima_forecast'] = arima_fit.forecast(len(test_data))[0]

# LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Define the model
model = Sequential()
model.add(LSTM(50, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# Reshape the data
train_X, train_y = train_data['point_value'].values.reshape(-1,1,1), train_data['point_value'].values.reshape(-1,1)
test_X, test_y = test_data['point_value'].values.reshape(-1,1,1), test_data['point_value'].values.reshape(-1,1)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2, validation_data=(test_X, test_y), callbacks=[early_stopping])

# Make predictions
lstm_pred = test_data.copy()
lstm_pred['lstm_forecast'] = model.predict(test_X)

# Prophet model
from prophet import Prophet
prophet_model = Prophet()
prophet_train = train_data.reset_index().rename(columns={'point_timestamp':'ds', 'point_value':'y'})
prophet_model.fit(prophet_train)
prophet_test = test_data.reset_index().rename(columns={'point_timestamp':'ds'})
prophet_pred = prophet_model.predict(prophet_test)['yhat']

# Step 5: Evaluate the performance of each model using different evaluation metrics

# ARIMA model
arima_mae = np.mean(abs(arima_pred['point_value'] - arima_pred['arima_forecast']))
arima_mse = np.mean((arima_pred['point_value'] - arima_pred['arima_forecast'])**2)
arima_rmse = np.sqrt(arima_mse)
arima_mape = np.mean(abs((arima_pred['point_value'] - arima_pred['arima_forecast'])/arima_pred['point_value'])) * 100

# LSTM model
lstm_mae = np.mean(abs(lstm_pred['point_value'] - lstm_pred['lstm_forecast']))
lstm_mse = np.mean((lstm_pred['point_value'] - lstm_pred['lstm_forecast'])**2)
lstm_rmse = np.sqrt(lstm_mse)
lstm_mape = np.mean(abs((lstm_pred['point_value'] - lstm_pred['lstm_forecast'])/lstm_pred['point_value'])) * 100

# Prophet model
prophet_mae = np.mean(abs(test_data['point_value'].values - prophet_pred))
prophet_mse = np.mean((test_data['point_value'].values - prophet_pred)**2)
prophet_rmse = np.sqrt(prophet_mse)
prophet_mape = np.mean(abs((test_data['point_value'].values - prophet_pred)/test_data['point_value'].values)) * 100

# Step 6: Compare the performance of each model using evaluation metrics
print('ARIMA Model Performance:')
print('MAE =', round(arima_mae, 2))
print('MSE =', round(arima_mse, 2))
print('RMSE =', round(arima_rmse, 2))
print('MAPE =', round(arima_mape, 2))
print('\n')

print('LSTM Model Performance:')
print('MAE =', round(lstm_mae, 2))
print('MSE =', round(lstm_mse, 2))
print('RMSE =', round(lstm_rmse, 2))
print('MAPE =', round(lstm_mape, 2))
print('\n')

print('Prophet Model Performance:')
print('MAE =', round(prophet_mae, 2))
print('MSE =', round(prophet_mse, 2))
print('RMSE =', round(prophet_rmse, 2))
print('MAPE =', round(prophet_mape, 2))

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=train_data.index, y=train_data['point_value'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=test_data.index, y=test_data['point_value'], mode='lines', name='Test'))
fig.add_trace(go.Scatter(x=arima_pred.index, y=arima_pred['arima_forecast'], mode='lines', name='ARIMA'))
fig.add_trace(go.Scatter(x=lstm_pred.index, y=lstm_pred['lstm_forecast'], mode='lines', name='LSTM'))
fig.add_trace(go.Scatter(x=prophet_test['ds'], y=prophet_pred, mode='lines', name='Prophet'))

fig.update_layout(title='Comparison of Models', xaxis_title='Date', yaxis_title='Point Value')
fig.show()

df = pd.read_csv("sample_1.csv")
df = df.iloc[: , 1:]

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Set up the data
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df.set_index('point_timestamp', inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ARIMA model
arima_model = ARIMA(train['point_value'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()
arima_predictions = arima_model_fit.forecast(len(test))
arima_mape = mean_absolute_percentage_error(test['point_value'], arima_predictions)

# LSTM model
def create_dataset(dataset, lookback=1):
    X, Y = [], []
    for i in range(len(dataset)-lookback):
        X.append(dataset[i:(i+lookback), 0])
        Y.append(dataset[i+lookback, 0])
    return np.array(X), np.array(Y)

lookback = 7
train_scaled = train.values.reshape(-1, 1)
test_scaled = test.values.reshape(-1, 1)
X_train, Y_train = create_dataset(train_scaled, lookback)
X_test, Y_test = create_dataset(test_scaled, lookback)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = lstm_predictions.reshape(-1)
lstm_mape = mean_absolute_percentage_error(test['point_value'].iloc[lookback:], lstm_predictions)

# Prophet model
prophet_df = pd.DataFrame({'ds': train.index, 'y': train['point_value']})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_predictions = prophet_model.predict(future)['yhat'][train_size:]
prophet_mape = mean_absolute_percentage_error(test['point_value'], prophet_predictions)

# Compare the models based on MAPE
mape_df = pd.DataFrame({
    'Model': ['ARIMA', 'LSTM', 'Prophet'],
    'MAPE': [arima_mape, lstm_mape, prophet_mape]
})
mape_df = mape_df.sort_values(by=['MAPE'])
print(mape_df)

df = pd.read_csv("sample_1.csv")
df = df.iloc[: , 1:]

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Set up the data
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df.set_index('point_timestamp', inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ARIMA model
arima_model = ARIMA(train['point_value'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()
arima_predictions = arima_model_fit.forecast(len(test))
arima_mape = mean_absolute_percentage_error(test['point_value'], arima_predictions)

# LSTM model
def create_dataset(dataset, lookback=1):
    X, Y = [], []
    for i in range(len(dataset)-lookback):
        X.append(dataset[i:(i+lookback), 0])
        Y.append(dataset[i+lookback, 0])
    return np.array(X), np.array(Y)

lookback = 7
train_scaled = train.values.reshape(-1, 1)
test_scaled = test.values.reshape(-1, 1)
X_train, Y_train = create_dataset(train_scaled, lookback)
X_test, Y_test = create_dataset(test_scaled, lookback)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = lstm_predictions.reshape(-1)
lstm_mape = mean_absolute_percentage_error(test['point_value'].iloc[lookback:], lstm_predictions)

# Prophet model
prophet_df = pd.DataFrame({'ds': train.index, 'y': train['point_value']})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_predictions = prophet_model.predict(future)['yhat'][train_size:]
prophet_mape = mean_absolute_percentage_error(test['point_value'], prophet_predictions)

# Classify the models based on MAPE
models = {
    'ARIMA': arima_mape,
    'LSTM': lstm_mape,
    'Prophet': prophet_mape
}
sorted_models = sorted(models.items(), key=lambda x: x[1])
if sorted_models[0][1] < 5:
    best_model = sorted_models[0][0]
    good_model = sorted_models[1][0]
    poor_model = sorted_models[2][0]
elif sorted_models[0][1] < 10:
    best_model = sorted_models[0][0]
    good_model = sorted_models[1][0]
    poor_model = sorted_models[2][0]
else:
    best_model = None
    good_model = None
    poor_model = None

#Print the results
print(f"Best model: {best_model}")
print(f"Good model: {good_model}")
print(f"Poor model: {poor_model}")

