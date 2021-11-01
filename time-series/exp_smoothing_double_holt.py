import pandas as pd
from statsmodels.tsa.holtwinters import Holt

df = pd.read_csv('data.csv')
y_label = 'value'
date_col = 'date'

df.set_index(date_col, inplace=True)
df.index = pd.to_datetime(df[date_col])
df = df.resample("D").sum()


# the forecast is currently made for the last n number of ovservations from the known data for evaluation purposes
forecast_length = 15


# train-test split
train = df.iloc[:-forecast_length]
test = df.iloc[-forecast_length:]


# define metric
def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100


# model
model_double = Holt(np.asarray(train[y_label]))
model_double._index = pd.to_datetime(train.index)

fit1_double = model_double.fit(smoothing_level=.3, smoothing_trend=.05)
fit2_double = model_double.fit(optimized=True)
fit3_double = model_double.fit(smoothing_level=.3, smoothing_trend=.2)


# make predictions
pred1_double = fit1_double.forecast(forecast_length)
pred2_double = fit2_double.forecast(forecast_length)
pred3_double = fit3_double.forecast(forecast_length)


# plot the results
import plotly.graph_objects as go
from plotly.offline import plot

for p, f, c in zip((pred1_double, pred2_double, pred3_double),(fit1_double, fit2_double, fit3_double),('coral','yellow','cyan')):
    fig_exp_smoothing_double.add_trace(go.Scatter(
        x=train.index, y=f.fittedvalues, marker_color=c, mode='lines',
        name=f"alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]}")
    )
    fig_exp_smoothing_double.add_trace(go.Scatter(
        x=pd.date_range(start=test.index.min(), periods=len(test) + len(p)),
        y=p, marker_color=c, mode='lines', showlegend=False)
    )
    print(f"\nMean absolute percentage error: {mape(test[y_label].values,p).round(2)} (alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]})")

plot(fig_exp_smoothing_double)