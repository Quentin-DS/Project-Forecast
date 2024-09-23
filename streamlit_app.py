import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
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

#import functions from the forecast model
from forecast_functions import prophet_best, evaluate_prophet, pmdarima_best, linear_reg_best, dummy_best, select_best_model
from forecast_functions import prophet_forecast, pmarima_forecast, linearreg_forecast, dummy_forecast
from forecast_functions import linear_reg_run, prophet_run, pmdarima_run, dummy_run

import warnings
warnings.filterwarnings('ignore')

import io


####################################################

df_test = pd.read_excel("top_ref_hn_streamlit.xlsx")
df_test["ds"] = pd.to_datetime(df_test["ds"],format = "%Y-%m-%d")
df_total_test = df_test.groupby("ds")["y"].sum()
df_total_test = pd.DataFrame(data = df_total_test)
df_total_test = df_total_test.reset_index()
model_results_test = pd.read_csv("model_filtered_streamlit.csv", index_col=0)
forecast_test = pd.read_excel("forecast_by_item_streamlit.xlsx")
forecast_test["ds"] = pd.to_datetime(forecast_test["ds"],format = "%Y-%m-%d")
forecast_total_test = pd.read_excel("forecast_total_streamlit.xlsx")
item_list_test = df_test["item"].unique()

st.title("Forecasting Tool per Item")
st.sidebar.title("Summary")
pages=["Project Introduction", "Data Loading", "Data Processing", "DataVizualization", "Forecasting Methodology & Running", "Forecasting Results", "Next Steps"]
page=st.sidebar.radio("Aller vers", pages)


############## PAGE 0 - Project Introduction #################

if page == pages[0] : 
  st.write("""
Welcome to our Sales Forecasting Tool! 
This application is designed to help organizations better understand and predict future sales trends for individual items.

## Purpose

Our Sales Forecasting Tool serves several crucial purposes for organizations:

 - Predictive Insights: By analyzing historical sales data, our tool generates forecasts for individual items. This allows businesses to anticipate future demand and trends with greater accuracy.

 - Inventory Optimization: With item-specific sales predictions, organizations can optimize their inventory levels. This helps prevent overstocking or stockouts, leading to improved cash flow and customer satisfaction.

 - Strategic Planning: Accurate forecasts enable better strategic decision-making : allocate resources, and set realistic sales targets based on data-driven predictions.

 - Cost Reduction: Improved forecasting can lead to significant cost savings through better supply chain management and reduced waste.


## How to Use

Navigate through the following pages to explore the full functionality of our tool:

 - Data Loading: Upload your sales data
 - Data Processing: Cleaning and alerting on your data for analysis
 - Data Visualization: Explore your data through charts and graphs
 - Forecasting Results: View and interpret sales predictions for each item

Start your journey to data-driven decision making by exploring each section!
""")
  
############## PAGE 1 - Data Loading #################

if page == pages[1] : 
  st.write("""### Data Loading
  In order to use the forecast tool, you need to load an EXCEL file having the same structure than the example shown below.
  
  Be careful with:
   - the format of the date (YYY-MM-DD)
   - the title of the columns
   - Sales has to be per month""")

  st.dataframe(df_test.head(10))
  
  st.write("### Load your file")
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.session_state.test = 0 #if we load a data set
  else:
    df = df_test
    st.session_state.test = 1 #if we don't load a data set
  
  df["ds"] = pd.to_datetime(df["ds"],format = "%Y-%m-%d")
  st.session_state.df = df
  

############## PAGE 2 - Data Processing #################

if page == pages[2] : 
  df = st.session_state.df
  df_positive = df[df["y"] >= 0]
  df_negative = df[df["y"] < 0]
  start=df_positive['ds'].min()
  end=df_positive['ds'].max()

  st.write("We are going to check the quality of the data.")
  st.write(f"The sales period is between the {start} and the {end}.")
  st.write("""Step1: we replace the negative value by 0, understanding that negative value in sales are returns.
           If you need to manually control those negative values, refer to the following dataframe.""")

  st.write("Dataframe with the negative value corrected.")
  if df_negative["y"].count() != 0:
    st.dataframe(df_negative)
  
  st.write("Step2: For each item we need to have all the months of the period, we input sales = 0 for months without sales for a given item.")
  #List of item
  item_list = df_positive["item"].unique()

  #creation of a Df with all months and all item, value 0 when no sales in order to have forecast for item with low sales
  all_months = pd.date_range(start=start, end=end, freq='MS')

  # Create a MultiIndex with all combinations of items and months
  index = pd.MultiIndex.from_product([item_list, all_months], names=['item', 'ds'])

  # Pivot the table to have Items as rows and Months as columns
  df_pivot = df_positive.pivot_table(index=['item', 'ds'], values='y', aggfunc='sum').reindex(index).fillna(0).reset_index()
  st.session_state.df = df_pivot

  st.write("Short analysis of the data to be forecasted:")
  st.dataframe(df_pivot.describe())
  st.write("""
  - count: number of months in the dataset
  - mean for y: mean of the sales per month per item
  - min for ds: first month of the data set
  - min for y: minimum value of sales
  - max for ds: last month of the data set
  - max for y: maximum value of sales
  - 25%/50%/75% for y: distribution of the value of sales in the dataset
  - std for y: standard deviation""")
  

############## PAGE 3 - DataVisualization #################

if page == pages[3] : 
  st.write("## Data Visualization")

  df = st.session_state.df
  df["year"] = df["ds"].dt.year
  df_total = df.groupby("ds")["y"].sum()

  df_item = df.groupby(["ds", "item"])["y"].sum()
  df_item = pd.DataFrame(data = df_item)
  df_item = df_item.reset_index()

  df_time = df.groupby("ds")["y"].sum()
  df_year = df
  df = df.drop("year", axis = 1)

  st.write("#### Graph of the total sales inputs.")
  st.session_state.df = df
  fig = plt.figure()
  sns.lineplot(data = df_total, x= df_total.index, y= df_total.values, errorbar=None,
                   label= "Total sales per month", color = "blue")
  plt.xlabel("Time")
  plt.ylabel("KG")
  plt.title("Total sales to forecast")
  st.pyplot(fig)

  st.write("#### Graph of the sales per item, with a filter to select the item to represent.")
  item_list = df["item"].unique()
  item_selected = st.multiselect("Select item(s)", item_list)
  fig = plt.figure()
  data = df_item[df_item["item"].isin(item_selected)]
  sns.lineplot(data = data, x= "ds", y= "y", errorbar=None,
                   color = "blue", hue = "item")
  plt.xlabel("Time")
  plt.ylabel("KG")
  plt.title("Total sales to forecast per item")
  st.pyplot(fig)

  st.write("#### Distribution of the sales per year.")
  fig = plt.figure()
  sns.boxplot(df_year, x = "year", y="y", color="blue")
  plt.ylabel("KG")
  plt.title("Volume distribution per year")
  st.pyplot(fig)

  st.write("#### Time Series decomposition")
  res = seasonal_decompose(df_time)
  fig = res.plot()
  st.pyplot(fig)
  st.write("Trend: The long-term pattern of the data, either upward, downward, or flat.")
  st.write("Seasonal: Regular patterns that repeat over time, like monthly or yearly cycles.")
  st.write("Residu: The part of the data that cannot be explained by the trend or seasonal components, considered random noise.")

############## PAGE 4 - Forecasting Methodology & Running #################

if page == pages[4] : 

  period = 12 #to simplify the model, then use a slicer to let the user choose
  st.session_state.period = period

  st.write("""## Generate the Forecast
  
  ### Methodology

  Forecasting model total sales:
  We will calculate the Forecast on the total sales, and return the MAPE. 
           To do this, we will use the Prophet forecasting model and perform a cross validation to find the parameters that result in the lower MAPE on the given period (12 months).
    
  Forecasting model by item:
   1. Run 4 models on each item and save the MAPE results and the parameters: Prophet, SARIMA, Linear Regression, Dummy
  *The test period is the end period of the dataset, respecting the length of the forecasting period (12 months).
   2. For Prophet and SARIMA, perform a cross validation to optimize the parameters in order to lower the MAPE.
   3. For each item, compare the MAPE results and select the best model.
   4. Finally run the forecast on the given period and concatenate all the result in an excel file.
   Important: in order to be align with a manufacturing strategy, the MAPE is calculated on the total period, not each month. The manufacturing production planning is relevant on a yearly bases in order to plan ressources.
           
  Formula is: MAPE = ABS(sum(Forecast) / sum(Test Period) - 1) *100""")

  if st.session_state.test == 0:
    
    df = st.session_state.df
    max_period = df["ds"].max()
  
    #Setting the df for the total sales analysis with Prophet
    df_total = df.groupby("ds")["y"].sum()
    df_total = pd.DataFrame(data = df_total)
    df_total = df_total.reset_index()
    st.session_state.df_total = df_total
 
    item_list = df["item"].unique()

    mcmc = 0 #to simplify the model, then use a slicer to let the user choose

  #with st.form(key='forecast_form_total'): #for slicer, not useful for the moment

    #st.write("### Forecast on total sales")
    #st.write("Step 1, test several period to test the model and find the best result (lower mean absolute percentage error).")
    #submit_button = st.form_submit_button(label='Generate forecast on total sales')
  
  #if submit_button:

    prophet_params, prophet_mape = prophet_best(df_total, period)
    m = Prophet(**prophet_params, mcmc_samples=mcmc).fit(df_total)
    future = m.make_future_dataframe(periods=period, freq='MS')
    forecast = m.predict(future)
 
    forecast["yhat"] = forecast["yhat"].clip(lower=0) #replace negative prediction by 0
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0) #replace negative prediction by 0
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0) #replace negative prediction by 0

    st.session_state.prophet_mape = prophet_mape
    st.session_state.forecasttotal = forecast


  ########Forecast by item using the models developed########
  #Forecast by item with Prophet, tunning Parameters, calculating MAPE
  #with st.form(key='forecast_form_item'): not useful without button

    #submit_button = st.form_submit_button(label='Generate forecast by item.')
 
 #if submit_button:
 
   #run the function best on each item via the function XXX_run
    model_results_prophet = prophet_run(df, period, item_list)
    model_results_sarima = pmdarima_run(df, period, item_list)
    model_results_lr = linear_reg_run(df, period, item_list)
    model_results_du = dummy_run(df, period, item_list)


  #Group the results in the same Df
    model_results = pd.concat([model_results_lr, model_results_prophet, model_results_sarima, model_results_du], axis = 0, ignore_index=True)

  #for each item, keep the best model from MAPE result, if MAPE = 0, dummy used
  # Group by 'item' and apply the selection function
    model_results_filtered = model_results.groupby('item').apply(select_best_model).reset_index(drop=True)
    
  #in the table filtered, input a flag to alert on MAPE > 30%, in order to have a business control later in the forecast results.
    model_results_filtered["business_control"] = np.where(model_results_filtered["MAPE"] > 30, "forecast to be controlled","ok")
    st.session_state.model_result = model_results_filtered
  
  ##Now forecasting each item on the period
  # Create a list to store all forecasts
    forecasts = []

  # Create a list of the next months on the period
    future_months = pd.date_range(start=max_period + timedelta(days=1), periods=period, freq='MS')

    for item in item_list:
        df_model = model_results_filtered[model_results_filtered["item"] == item].iloc[0]
        item_data = df[df["item"] == item]
    
        if df_model["model"] == "Prophet":
            prediction = prophet_forecast(item_data, df_model["best_params"], period, mcmc)
        elif df_model["model"] == "SARIMA":
            prediction = pmarima_forecast(item_data, df_model["best_params"], period)
        elif df_model["model"] == "LinearRegression":
            prediction = linearreg_forecast(item_data, period)
        else:
            prediction = dummy_forecast(item_data, period)
    
      # Create a DataFrame for this item's forecast
        item_forecast = pd.DataFrame({
            'item': [item] * period,
            'ds': future_months,
            'yhat': prediction
        })
    
        forecasts.append(item_forecast)

  # Combine all forecasts into a single DataFrame
    forecast_item = pd.concat(forecasts, ignore_index=True)

  # Sort the DataFrame by item and month
    forecast_item = forecast_item.sort_values(['item', 'ds'])

  # Reset the index
    forecast_item = forecast_item.reset_index(drop=True)

  #input the business comment in the forecast table
    forecast_item = forecast_item.merge(model_results_filtered[["item", "business_control"]], on = ["item"], how = "left")
    st.session_state.forecast_item = forecast_item


############## PAGE 5 - Forecasting Results #################

if page == pages[5]:
    period = st.session_state.period
    st.write("### Forecast on total sales")

    if st.session_state.test == 0:

      ###### Forecast total#######
      forecast = st.session_state.forecasttotal
      df = st.session_state.df
      max_period = df["ds"].max()
      df_total = st.session_state.df_total

      fig, ax = plt.subplots(figsize=(12,6))
      df_total.plot(x = "ds", y = "y", label = "Actual data", color = "blue", ax=ax)
      forecast[forecast["ds"] > max_period].plot(x="ds", y="yhat", color = "green", label = "Forecast", ax=ax)
      forecast[forecast["ds"] > max_period].plot(x="ds", y="yhat_upper", color = 'black', label = "Fcst upper limit", linestyle='--', ax =ax)
      forecast[forecast["ds"] > max_period].plot(x="ds", y="yhat_lower", color = 'black', label = "Fcst lower limit", linestyle='--', ax=ax)
      plt.axvline(x=df_total["ds"].max(), color='red', linestyle='--', label='Forecast Start')
      plt.xlabel("Time")
      plt.ylabel("KG")
      plt.legend()
      plt.title("Forecast from Prophet model optimized.")
      st.pyplot(fig)

      prophet_mape = st.session_state.prophet_mape
      st.write(f"The Mean Absolute Percentage Error on the test period ({period} months) is {round(prophet_mape,2)}%")

      # Generate the Excel data in memory
      buffer = io.BytesIO()
      forecast.to_excel(buffer, index=False)

  # Reset the buffer position to the beginning
      buffer.seek(0)

  #exportation of the file
      st.download_button(label="Download Forecast total", data=buffer, file_name='forecast_total.xlsx')

      ###### Forecast item #######
      st.write("""### Forecast per item
               
    Outputs:
     - Graph by item (to select) with historical data and forecast.
     - Excel file with forecast by month.
     - Dataframe with model selected for each item, the optimized parameters, the MAPE and the business comments.""")
      
      forecast_item = st.session_state.forecast_item
      model_results_filtered = st.session_state.model_result

       #show best model selected by item
      st.dataframe(model_results_filtered)

  # Generate the Excel data in memory
      buffer = io.BytesIO()
      forecast_item.to_excel(buffer, index=False)

  # Reset the buffer position to the beginning
      buffer.seek(0)

  #exportation of the file
      st.download_button(label="Download Forecast by item", data=buffer, file_name='forecast_item.xlsx')

  ## Graph by item, box to select the item, and the graph is updated.
      st.write("#### Graph of the forecasts per item, with a filter to select the item to represent.")
      item_list = df["item"].unique()
      item_selected = st.selectbox("Select 1 item", item_list)
      fig, ax = plt.subplots(figsize=(12,6))

      data = df[df["item"] == item_selected]
      data.plot(x= "ds", y= "y", color = "blue", ax=ax)

      forecast_item[forecast_item["item"] == item_selected].plot(x = "ds", y = "yhat", color = "green", ax = ax)
      plt.axvline(x=max_period, color='red', linestyle='--', label='Forecast Start')
      plt.legend()
      plt.xlabel("Time")
      plt.ylabel("KG")
      plt.title("Total sales to forecast per item")
      st.pyplot(fig)

  #MAPE calculation, taking the average of the item selected
      MAPE_item = model_results_filtered[model_results_filtered["item"] == item_selected]["MAPE"].values[0]
      st.write(f"The Mean Absolute Percentage Error on the test period ({period} months) for the selected item(s) is {round(MAPE_item,2)}%")



    else:
      ###### Forecast total test #######
      max_period = df_test["ds"].max()
      fig, ax = plt.subplots(figsize=(12,6))
      df_total_test.plot(x = "ds", y = "y", label = "Actual data", color = "blue", ax=ax)
      forecast_total_test[forecast_total_test["ds"] > max_period].plot(x="ds", y="yhat", color = "green", label = "Forecast", ax=ax)
      forecast_total_test[forecast_total_test["ds"] > max_period].plot(x="ds", y="yhat_upper", color = 'black', label = "Fcst upper limit", linestyle='--', ax =ax)
      forecast_total_test[forecast_total_test["ds"] > max_period].plot(x="ds", y="yhat_lower", color = 'black', label = "Fcst lower limit", linestyle='--', ax=ax)
      plt.axvline(x=df_total_test["ds"].max(), color='red', linestyle='--', label='Forecast Start')
      plt.xlabel("Time")
      plt.ylabel("KG")
      plt.legend()
      plt.title("Forecast from Prophet model optimized.")
      st.pyplot(fig)
      st.write(f"The Mean Absolute Percentage Error on the test period (12 months) is 19.79%")

      ###### Forecast item test #######
      st.write("""### Forecast per item
               
    Outputs:
     - Graph by item (to select) with historical data and forecast.
     - Excel file with forecast by month.
     - Dataframe with model selected for each item, the optimized parameters, the MAPE and the business comments.""")
      
      #show best model selected by item
      st.dataframe(model_results_test)
      
      ## Graph by item, box to select the item, and the graph is updated.
      st.write("#### Graph of the forecasts per item, with a filter to select the item to represent.")
      item_selected = st.selectbox("Select 1 item", item_list_test)
      fig, ax = plt.subplots(figsize=(12,6))

      data = df_test[df_test["item"] == item_selected].sort_values(by="ds")
      data.plot(x= "ds", y= "y", label = "Data", color = "blue", ax=ax)

      forecast_test[forecast_test["item"] == item_selected].plot(x = "ds", y = "yhat", label = "Forecast", color = "green", ax = ax)
      plt.axvline(x=max_period, color='red', linestyle='--', label='Forecast Start')
      plt.legend()
      plt.xlabel("Time")
      plt.ylabel("KG")
      plt.title("Total sales to forecast per item")
      st.pyplot(fig)

  #MAPE calculation, taking the MAPE of the item selected
      MAPE_item = model_results_test[model_results_test["item"] == item_selected]["MAPE"].values[0]
      model_item = model_results_test[model_results_test["item"] == item_selected]["model"].values[0]
      st.write(f"The Mean Absolute Percentage Error on the test period ({period} months) for the selected item is {round(MAPE_item,2)}% using the model {model_item}.")


############## PAGE 6 - Next steps #################

if page == pages[6]:
  st.write("""To enhance model robustness the next steps should be:
  - incorporate manual input capabilities for contracted sales.
  - the auto ARIMA model's performance can be potentially improved by expanding parameter space while employing scikit-optimize to mitigate computational overhead.
  - allow the user to select the forecasting period.
  - allow the user to select the MAPE calculation: full period or by month.
  - connect a LLM to improve the user experience ?
  - add feedbacks from users ?""")
