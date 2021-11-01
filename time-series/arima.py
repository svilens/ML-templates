import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('data.csv')
y_label = 'value'
date_col = 'date'
rolling_avg_window = 7

df.set_index(date_col, inplace=True)
df.index = pd.to_datetime(df[date_col])
df = df.resample("D").sum()


# the forecast is currently made for the last n number of ovservations from the known data for evaluation purposes
forecast_length = 15


# ARIMA works on stationary data, so rolling average is needed
df2 = df.copy()
df2[y_label] = df[y_label].rolling(window=rolling_avg_window,center=False).mean().dropna()


# train-test split
train = df2.iloc[:-forecast_length]
test = df2.iloc[-forecast_length:]


# define metric
def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100


# define ARIMA function
def arima_province(train, test, forecast_length):
    p=d=q=range(0,7)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(train, order=var)
            result = model.fit(disp=0)
            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
        
    #Modeling
    model = ARIMA(train, order=param)
    result = model.fit(disp=0)
    
    pred = result.forecast(steps=len(test))[0]
    #Printing the error metrics
    model_error = mape(test, pred)

    return pred, model_error


# make predictions
pred, model_error = arima_province(train, test, forecast_length)