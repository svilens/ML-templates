import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES


df = pd.read_csv('data.csv')
y_label = 'value'
date_col = 'date'

df.set_index(date_col, inplace=True)
df.index = pd.to_datetime(df[date_col])
df = df.resample("D").sum()


# the forecast is currently made for the last n number of ovservations from the known data for evaluation purposes
forecast_length = 15
seasonal_periods = 7


# train-test split
train = df.iloc[:-forecast_length]
test = df.iloc[-forecast_length:]

train[y_label] = train[y_label].replace(0,0.1)


# define metric
def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100


# model
model_triple = HWES(train[y_label], seasonal_periods=seasonal_periods, trend='add', seasonal='mul')
model_triple = model_triple.fit(optimized=True, use_brute=True)

#print out the training summary
print(model_triple.summary())


# make predictions
pred_triple = model_triple.forecast(steps=forecast_length)


# plot the results
import plotly.graph_objects as go
from plotly.offline import plot

fig_exp_smoothing_triple = go.Figure()
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=train.index, y=train[y_label], name='Historical data', mode='lines')
)
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=train.index, y=model_triple.fittedvalues, name='Model fit', mode='lines', marker_color='lime')
)
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=test.index, y=test[y_label], name='Validation data', mode='lines', marker_color='coral')
)
fig_exp_smoothing_triple.add_trace(go.Scatter(
    x=pd.date_range(start=test.index.min(), periods=len(test) + len(pred_triple)),
    y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
)

plot(fig_exp_smoothing_triple)