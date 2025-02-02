# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:32:50 2024

extract_data.py
This script processes e-commerce customer data for analytical and predictive modeling purposes. It starts by loading a raw dataset, cleans the data by handling missing values and 
removing unnecessary columns, and then calculates important business metrics such as Customer Lifetime Value (CLV), Churn Rate, and Conversion Rate. These metrics are essential for 
evaluating customer behavior and business performance. Finally, the cleaned dataset with calculated metrics is saved for further analysis and use in machine learning models.

Steps in the script:

Load Data: Load the raw dataset from a CSV file.
Clean Data: Handle missing values, remove unnecessary or sensitive columns, and convert data types as needed (e.g., dates).
Calculate Metrics: Compute key business metrics like CLV, churn rate, and conversion rate.
Save Cleaned Data: Save the cleaned and processed dataset to a new CSV file, ready for use in further analysis or predictive modeling.

@author: Carlos A. Duran Viilalobos
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Clean the dataset by filling missing values, removing unnecessary columns, and converting data types.
    """
    # Handle missing values in 'Returns' by filling them with 0
    df = df.fillna({'Returns': 0})
    
    # Drop the duplicate 'Age' column and the 'Customer Name' column to remove sensitive information
    df.drop(columns=['Age', 'Customer Name'], inplace=True)
    
    # Convert 'Purchase Date' to datetime format
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    
    return df

def calculate_metrics(df):
    """
    Calculate key metrics such as Customer Lifetime Value (CLV), Churn Rate, and Conversion Rate.
    """
    # Calculate total purchase amount by customer
    total_purchase_amount = df.groupby('Customer ID')['Total Purchase Amount'].sum()
    
    # Calculate total returns by customer
    total_returns = df.groupby('Customer ID')['Returns'].sum()
    
    # Calculate net customer revenue by subtracting returns from total purchase amount
    customer_revenue = total_purchase_amount - total_returns
    
    # Calculate Purchase Frequency by Customer
    customer_frequency = 1 #df.groupby('Customer ID').size() # Since we have the sum revenue over 3 years
    
    # Assuming an average customer lifespan (can be customized)
    average_customer_lifespan = 3  # in years
    
    # Calculate Customer Lifetime Value (CLV)
    clv = customer_revenue * customer_frequency * average_customer_lifespan
    
    # Map the calculated CLV back to the original dataframe
    df['Customer Lifetime Value (CLV)'] = df['Customer ID'].map(clv)
    
    # Calculate Churn Rate (Churn = 1 means churned, 0 means active)
    churn_rate = df['Churn'].mean() * 100
    
    # Calculate Conversion Rate
    total_visits = df['Quantity'].sum()  # Assuming 'Quantity' represents interactions/visits
    total_conversions = df[df['Total Purchase Amount'] > 0].shape[0]
    conversion_rate = (total_conversions / total_visits) * 100
    
    # Print calculated metrics
    print(f"Average Churn Rate: {churn_rate:.2f}%")
    print(f"Conversion Rate: {conversion_rate:.2f}%")
    
    return df

def main():
    # Path to the dataset
    filepath = '../data/ecommerce_customer_data_large.csv'
    
    # Load the data
    df_large = load_data(filepath)
    
    # Clean the data
    df_large = clean_data(df_large)
    
    # Calculate metrics
    df_large = calculate_metrics(df_large)
    
    # Save the cleaned dataset with calculated metrics
    df_large.to_csv('../data/cleaned_ecommerce_customer_data_with_metrics.csv', index=False)
    
    print("Data preparation and metric calculation complete. Cleaned data with metrics saved to 'data/cleaned_ecommerce_customer_data_with_metrics.csv'.")

if __name__ == "__main__":
    main()
