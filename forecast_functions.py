import pandas as pd
import numpy as np

#Forecasting Models
import pmdarima as pm
from prophet import Prophet
from sklearn.linear_model import LinearRegression

#MultiProcessing and Optimization for Prophet cross validation
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

#for cross validation analysis Prophet
#import itertools
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')



#function Prophet and MAPE calculation

def evaluate_prophet(params, df_train, df_test, period):
    m = Prophet(**params).fit(df_train)
    future = m.make_future_dataframe(periods=period, freq='MS')
    forecast = m.predict(future)
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)
    
    prediction = forecast["yhat"].iloc[-period:].reset_index(drop=True)
    test = df_test["y"].reset_index(drop=True)
    
    #mape = np.mean(np.abs((test - prediction) / (test + 0.01)) * 100)
    mape = np.abs(sum(prediction)/sum(test) - 1) * 100
    return mape

def prophet_best(df, period, n_calls=100):
    train_period = df["ds"].sort_values().unique()[:-period]
    test_period = df["ds"].sort_values().unique()[-period:]

    df_train = df[df["ds"].isin(train_period)]
    df_test = df[df["ds"].isin(test_period)]

    space = [
        Real(0.01, 1, "log-uniform", name='changepoint_prior_scale'),
        Real(0.01, 1, "log-uniform", name='seasonality_prior_scale'),
        Real(0.7, 1, "log-uniform", name="changepoint_range"),
        #Integer(1, 12, name="yearly_seasonality"),
        Categorical(['multiplicative', 'additive'], name="seasonality_mode"),
        Categorical(["linear", "flat"], name="growth"),
        ]

    @use_named_args(space)
    def objective(**params):
        return evaluate_prophet(params, df_train, df_test, period)

    with ProcessPoolExecutor() as executor:
        res_gp = gp_minimize(objective, space, n_calls=n_calls, n_jobs=-1)

    best_params = {dim.name: value for dim, value in zip(space, res_gp.x)}
    best_mape = res_gp.fun

    return best_params, best_mape

#function Auto SARIMA and MAPE calculation for multiplicative time series

def pmdarima_best(df, period):

    train_period = df["ds"].sort_values(ascending=True).unique()[:-period]
    test_period = df["ds"].sort_values(ascending=True).unique()[-period:]

    df_train = df[df["ds"].isin(train_period)]
    df_test = df[df["ds"].isin(test_period)]

    data = df_train.groupby("ds")["y"].sum()
    
    model = pm.auto_arima(data)
    
    #Calcul the MAPE
    prediction = model.predict(n_periods=period)
    prediction[prediction < 0] = 0 #replace negative prediction by 0
    test = df_test["y"]
    test = test.reset_index(drop=True)
    #mape = (np.abs((test - prediction.values) / (test+0.01)) * 100).mean(axis = 0)
    mape = np.abs(sum(prediction)/sum(test) - 1) * 100
    best_params = model.get_params()
    
    return (best_params, mape)

#function with LinearRegression, and MAPE calculation
def linear_reg_best(df, period):
    train_period = df["ds"].sort_values(ascending=True).unique()[:-period]
    test_period = df["ds"].sort_values(ascending=True).unique()[-period:]

    df_train = df[df["ds"].isin(train_period)]
    df_test = df[df["ds"].isin(test_period)]

    X_train = np.array(df_train.index).reshape(-1,1)
    y_train = df_train[["y"]]

    X_test = np.array(df_test.index).reshape(-1,1)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    prediction = np.where(prediction < 0, 0, prediction) #replace negative prediction by 0
    test = df_test["y"]
    test = test.reset_index(drop=True)
    #mape = (np.abs((test - prediction) / (test+0.01)) * 100).mean(axis = 0)
    mape = np.abs(sum(prediction)/sum(test) - 1) * 100

    return mape[0]

#function with dummy forecast, and MAPE calculation
def dummy_best(df, period):
    copy_period = df["ds"].sort_values(ascending=True).unique()[-period*2:-period]
    test_period = df["ds"].sort_values(ascending=True).unique()[-period:]

    df_copy = df[df["ds"].isin(copy_period)]
    df_test = df[df["ds"].isin(test_period)]

    prediction = df_copy["y"]
    prediction = prediction.reset_index(drop=True)
    test = df_test["y"]
    test = test.reset_index(drop=True)
    #mape = (np.abs((test - prediction) / (test+0.01)) * 100).mean(axis = 0)
    mape = np.abs(sum(prediction)/sum(test) - 1) * 100

    return mape

def select_best_model(group):
    # If there are any non-zero MAPE values, select the minimum
    if (group['MAPE'] != 0).any():
        return group.loc[group['MAPE'].idxmin()]
    
    # If all MAPE values are 0, prefer the Dummy model
    dummy_model = group[group['model'] == 'Dummy']
    if not dummy_model.empty:
        return dummy_model.iloc[0]
    
    # If no Dummy model, just return the first row
    return group.iloc[0]

###### Function to run the forcaste best on each item #####

def prophet_run(df, period, item_list):
    best_prophet = []
    best_mape_pro = []
    for item in item_list:
        prophet_params, prophet_mape = prophet_best(df[df["item"] == item], period)
        best_prophet.append(prophet_params)
        best_mape_pro.append(prophet_mape)
 
    model_results_prophet = pd.DataFrame({"item":item_list, 
                                     "model":"Prophet", 
                                     "best_params": best_prophet, 
                                     "MAPE": best_mape_pro})
    return model_results_prophet
  
def pmdarima_run(df, period, item_list):
    best_sarima = []
    best_mape_sar = []
    for item in item_list:
        sarima_params, sarima_mape = pmdarima_best(df[df["item"] == item], period)
        best_sarima.append(sarima_params)
        best_mape_sar.append(sarima_mape)

    model_results_sarima = pd.DataFrame({"item":item_list, 
                                     "model":"SARIMA", 
                                     "best_params": best_sarima, 
                                     "MAPE": best_mape_sar})
    return model_results_sarima

def linear_reg_run(df, period, item_list):  
    best_mape_lr = []
    for item in item_list:
        lr_mape = linear_reg_best(df[df["item"] == item], period)
        best_mape_lr.append(lr_mape)

    model_results_lr = pd.DataFrame({"item":item_list, 
                                  "model":"LinearRegression", 
                                  "best_params": "", 
                                  "MAPE": best_mape_lr})
    return model_results_lr

def dummy_run(df, period, item_list):
    best_mape_du = []
    for item in item_list:
        du_mape = dummy_best(df[df["item"] == item], period)
        best_mape_du.append(du_mape)

    model_results_du = pd.DataFrame({"item":item_list, 
                                  "model":"Dummy", 
                                  "best_params": "", 
                                  "MAPE": best_mape_du})
    return model_results_du


#functions to calculate forecast on a period, taking the best params as argument

def prophet_forecast(df, params, period, mcmc):
    m = Prophet(**params, mcmc_samples=mcmc).fit(df)
    future = m.make_future_dataframe(periods=period, freq='MS')
    forecast = m.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0) #replace negative prediction by 0
    prediction = forecast["yhat"].tail(period)
    
    return prediction

def pmarima_forecast(df, params, period):
    
    data = df.groupby("ds")["y"].sum()
    model = pm.arima.ARIMA(**params).fit(data)
    prediction = model.predict(n_periods=period)
    

    return prediction

def linearreg_forecast(df, period):
    
    X_train = np.array(df.index).reshape(-1,1)
    y_train = df["y"]

    X_test = np.array(range(len(X_train), len(X_train) + period)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    prediction = np.where(prediction < 0, 0, prediction)
    
    
    return prediction

def dummy_forecast(df, period):
    copy_period = df["ds"].sort_values(ascending=True).unique()[-period:]
    
    df_copy = df[df["ds"].isin(copy_period)]

    prediction = df_copy["y"]
    prediction = prediction.reset_index(drop=True)
    
    return prediction