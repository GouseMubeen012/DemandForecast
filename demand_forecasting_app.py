# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# Step 1: Load and preprocess data
def load_data():
    customer_demographics = pd.read_csv('CustomerDemographics.csv')
    product_info = pd.read_csv('Productinfo.csv')
    transaction_data = pd.read_csv('Transactiondata1.csv')
    
    # Convert InvoiceDate to datetime
    transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
    
    return customer_demographics, product_info, transaction_data

# Step 2: Perform EDA
def perform_eda(customer_demographics, product_info, transaction_data):
    # Customer-level summary
    customer_summary = customer_demographics.groupby('Country').size().reset_index(name='CustomerCount')
    
    # Item-level summary
    item_summary = transaction_data.groupby('StockCode').agg({
        'Quantity': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    # Transaction-level summary
    transaction_summary = transaction_data.groupby('Invoice').agg({
        'Quantity': 'sum',
        'Price': 'sum'
    }).reset_index()
    
    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.histplot(transaction_data['Quantity'], bins=50)
    plt.title('Distribution of Quantity Sold')
    plt.xlabel('Quantity')
    plt.ylabel('Count')
    plt.show()
    
    return customer_summary, item_summary, transaction_summary

# Step 3: Identify top 10 stock codes and high revenue products
def get_top_products(transaction_data):
    # Top 10 by quantity sold
    top_10_quantity = transaction_data.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()
    
    # Top 10 by revenue
    transaction_data['Revenue'] = transaction_data['Quantity'] * transaction_data['Price']
    top_10_revenue = transaction_data.groupby('StockCode')['Revenue'].sum().nlargest(10).index.tolist()
    
    return top_10_quantity, top_10_revenue

# Step 4: Time Series Analysis
def prepare_time_series_data(transaction_data, stock_code):
    ts_data = transaction_data[transaction_data['StockCode'] == stock_code]
    ts_data = ts_data.groupby('InvoiceDate')['Quantity'].sum().resample('W').sum()
    return ts_data

def fit_arima(ts_data):
    model = ARIMA(ts_data, order=(1,1,1))
    results = model.fit()
    return results

def fit_ets(ts_data):
    model = ExponentialSmoothing(ts_data)
    results = model.fit()
    return results

def fit_prophet(ts_data):
    df = pd.DataFrame({'ds': ts_data.index, 'y': ts_data.values})
    model = Prophet()
    model.fit(df)
    return model

# Step 5: Non-Time Series Techniques
def prepare_ml_data(transaction_data, customer_demographics, product_info):
    # Merge data
    merged_data = transaction_data.merge(customer_demographics, on='Customer ID')
    merged_data = merged_data.merge(product_info, on='StockCode')
    
    # Feature engineering
    merged_data['WeekOfYear'] = merged_data['InvoiceDate'].dt.isocalendar().week
    merged_data['DayOfWeek'] = merged_data['InvoiceDate'].dt.dayofweek
    
    return merged_data

def fit_decision_tree(X, y):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model

def fit_xgboost(X, y):
    model = XGBRegressor(random_state=42)
    model.fit(X, y)
    return model

# Step 6: Training and Validation Strategy
def time_based_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        yield X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

# Step 7: Forecasting
def forecast(model, periods=15):
    return model.forecast(steps=periods)

# Step 8: Error and Evaluation Metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

# Step 9: ACF and PACF Plots
def plot_acf_pacf(ts_data):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(ts_data, ax=ax1)
    plot_pacf(ts_data, ax=ax2)
    plt.show()

# Step 10: Streamlit App
def create_app():
    st.title('Demand Forecasting System')
    
    # Load data
    customer_demographics, product_info, transaction_data = load_data()
    
    # Get top 10 products
    top_10_quantity, top_10_revenue = get_top_products(transaction_data)
    
    st.write("Top 10 Products by Quantity Sold:")
    st.write(top_10_quantity)
    
    st.write("Top 10 Products by Revenue:")
    st.write(top_10_revenue)
    
    # User input
    selected_stock_code = st.selectbox("Select a Stock Code", top_10_quantity)
    
    if selected_stock_code:
        # Prepare time series data
        ts_data = prepare_time_series_data(transaction_data, selected_stock_code)
        
        # Fit ARIMA model (you can add other models as well)
        model = fit_arima(ts_data)
        
        # Forecast
        forecast_values = forecast(model)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ts_data.plot(ax=ax, label='Historical')
        forecast_values.plot(ax=ax, label='Forecast')
        plt.title(f'Demand Forecast for Stock Code {selected_stock_code}')
        plt.legend()
        st.pyplot(fig)
        
        # Error histogram
        residuals = model.resid
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        plt.title('Error Distribution')
        st.pyplot(fig)
        
        # Download forecast as CSV
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=7), periods=15, freq='W'),
            'Forecasted_Demand': forecast_values
        })
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"forecast_{selected_stock_code}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    create_app()