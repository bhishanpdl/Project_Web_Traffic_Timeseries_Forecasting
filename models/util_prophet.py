
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') # better than sns styles.
matplotlib.rcParams['figure.figsize'] = 12,8

import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.tools import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode

from numba import jit
import math

# https://www.kaggle.com/cpmpml/smape-weirdness
@jit(nopython=True)
def get_smape(y_true, y_pred):
    A = y_true.to_numpy().ravel()
    F = y_pred.to_numpy().ravel()[:len(A)]
    return ( 200.0/len(A) * np.sum(  np.abs(F - A) / 
                                  (np.abs(A) + np.abs(F) + np.finfo(float).eps))
           )


# https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/37232
@jit(nopython=True)
def get_smape_fast(y_true, y_pred):
    """Fast implementation of SMAPE.
    
    Parameters
    -------------
    y_true: numpy array with no NaNs and non-negative
    y_pred: numpy array with no NaNs and non-negative
    
    Returns
    -------
    out : float
    """
    out = 0
    for i in range(y_true.shape[0]):
        if (y_true[i] != None and np.isnan(y_true[i]) ==  False):
            a = y_true[i]
            b = y_pred[i]
            c = a+b
            if c == 0:
                continue
            out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out

def safe_median(s):
    return np.median([x for x in s if ~np.isnan(x)])

def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_actual_forecast_mpl(df, forecast_str_lst, forecast_lst):
    """Plot prophet forcast.
    
    Parameters
    -----------
    df -- dataframe with columns ds,y (cap and floor are optional)
    forecast_str_lst -- list of strings
    forecast_lst -- list of forecasts
    
    Example
    --------
    forecast_str_lst = ['forecast1', 'forecast2', 'forecast3','forecast4']
    forecast_lst = [eval(i) for i in forecast_str_lst]
    plot_actual_forecast_mpl(df, ['forecast1'], [forecast1])
    
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    

    plt.figure(figsize=(12,8))
    plt.plot(df.y,'b',label='original')
    

    colors10_hex = ['#b03060','#ff0000', '#ff00ff',
                '#67ceab', '#63c56c', '#225e31',
                 '#29b6f6', '#6495ed','#00008b', 
                '#ffa500']

    for i,f in enumerate(forecast_str_lst):
        forecast = forecast_lst[i]
        plt.plot(forecast.yhat,c=colors10_hex[i],label=f)
        
    plt.legend()

def plot_actual_forecast_sns(df, forecast_str_lst, forecast_lst):
    """Plot prophet forcast.
    
    Parameters
    -----------
    df -- dataframe with columns ds,y (cap and floor are optional)
    forecast_str_lst -- list of strings
    
    Example
    --------
    forecast_str_lst = ['forecast1', 'forecast2', 'forecast3','forecast4']
    forecast_lst = [eval(i) for i in forecast_str_lst]
    plot_actual_forecast_sns(df2, forecast_str_lst, forecast_lst)
    
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    

    plt.figure(figsize=(12,8))
    
    df_plot = df2[['y']]
    df_plot.index = pd.to_datetime(df2['ds'])

    for i,f in enumerate(forecast_str_lst):
        forecast = forecast_lst[i]
        ts = forecast.yhat
        ts.index = pd.to_datetime(forecast.ds)
        df_tmp = pd.DataFrame({f: ts})
        df_plot = pd.concat([df_plot,df_tmp], axis=1)


    sns.lineplot(data=df_plot)

def plot_actual_forecast_plotly(df, forecast_str_lst, forecast_lst):
    """Plot prophet forcast.
    
    Parameters
    -----------
    df -- dataframe with columns ds,y (cap and floor are optional)
    forecast_str_lst -- list of strings
    forecast_lst -- list of forecasts
    
    Example
    --------
    forecast_str_lst = ['forecast1', 'forecast2', 'forecast3','forecast4']
    forecast_lst = [eval(i) for i in forecast_str_lst]
    plot_actual_forecast_plotly(df2, forecast_str_lst,forecast_lst)
    
    """
    from plotly.offline import plot, iplot, init_notebook_mode

    df_plot = df[['y']]
    df_plot.index = pd.to_datetime(df['ds'])

    for i,f in enumerate(forecast_str_lst):
        forecast = forecast_lst[i]
        ts = forecast.yhat
        ts.index = pd.to_datetime(forecast.ds)
        df_tmp = pd.DataFrame({f: ts})
        df_plot = pd.concat([df_plot,df_tmp], axis=1)


    iplot([{'x': df_plot.index,'y': df_plot[col],'name': col}  
           for col in df_plot.columns
          ])

def plot_deltas(m):
    """Plot model params delta as bar plots.
    
    Notes:
    -------
    1. If the barplot is all incresing downward,
       we may need to change these quantities:
       - changepoint_range=0.8 (default is 0.8)
       - changepoint_prior_scale=0.7 (default is 0.05)
    
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')


    deltas = m.params['delta'].mean(axis=0)

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111)

    ax.bar(range(len(deltas)), deltas)

    ax.set_ylabel('Rate change (delta)')
    ax.set_xlabel('Potential changepoint')
    fig.tight_layout()

def plot_deltas_plotly(m):
    """Plot prophet forecast params delta values.
    
    """
    import plotly.graph_objs as go
    from plotly.offline import plot, iplot
    
    # data to plot
    x = list(range(len(m.params['delta'][0])))
    y = m.params['delta'].ravel().tolist()
    
    # trace
    trace = go.Bar(x= x,y=y,name='Change Points')
    data = [trace]
    fig = go.Figure(data=data)
    iplot(fig)
    
def outliers_to_na(ts, devs):
    """Replace the outliers by na.
    
    Then we can again fill na by 0.
    Here, in this wikipedia data nans are given 0.
    
    
    """
    median= ts['y'].median()
    std = np.std(ts['y'])

    for x in range(len(ts)):
        val = ts['y'][x]

        if (val < median - devs * std or val > median + devs * std):
            ts['y'][x] = None 
    return ts

def convert_ts_to_prophet_df(ts):
    """Convert timeseries to dataframe required by prophet.
    
    Parameters
    -----------
    ts: timeseries with index as datetime and have values
    
    """
    df = pd.DataFrame(columns=['ds','y'])
    
    df['ds'] = pd.to_datetime(ts.index)
    df.index = df['ds']
    df['y'] = ts.to_numpy()

    return df

def remove_negs_from_forecast(forecast):
    """Replace negative forecasts by 0.
    
    Parameters
    ----------
    forecast -- dataframes returned by prophet
    
    Example
    --------

    m1 = Prophet()
    m1.fit(df1);
    
    future1 = m1.make_future_dataframe(periods=60)
    forecast1 = m1.predict(future1)
    """
    forecast = forecast.copy()
    forecast['yhat'] = forecast['yhat'].clip_lower(0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip_lower(0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip_lower(0)
    
    return forecast
