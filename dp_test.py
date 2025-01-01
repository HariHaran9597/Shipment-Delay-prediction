# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

def prepare_data(file_path):
    # Load the dataset (for .xlsx files)
    df = pd.read_excel(file_path)  # Change to pd.read_excel()

    # Inspect the dataset
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Handle missing values (drop rows with excessive nulls)
    df = df.dropna()  # Alternatively, impute missing values based on context

    # Convert date columns to datetime format
    date_cols = ["Shipment Date", "Planned Delivery Date", "Actual Delivery Date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Feature Engineering
    # Calculate 'Planned Delivery Duration' (difference between 'Planned Delivery Date' and 'Shipment Date')
    df['Planned Delivery Duration'] = (df['Planned Delivery Date'] - df['Shipment Date']).dt.days

    # Calculate 'Actual Delivery Delay' (difference between 'Actual Delivery Date' and 'Planned Delivery Date')
    df['Actual Delivery Delay'] = (df['Actual Delivery Date'] - df['Planned Delivery Date']).dt.days

    # Extract additional date-related features
    df['Shipment Month'] = df['Shipment Date'].dt.month
    df['Shipment Day of Week'] = df['Shipment Date'].dt.dayofweek
    df['Is Weekend'] = df['Shipment Day of Week'].apply(lambda x: 1 if x >= 5 else 0)

    # Encode categorical features
    categorical_cols = ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders for potential decoding later

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['Distance (km)', 'Planned Delivery Duration', 'Actual Delivery Delay']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure target variable is binary
    df['Delayed'] = df['Delayed'].apply(lambda x: 1 if x == "Yes" else 0)

    # Final dataset overview
    print("\nProcessed Dataset Overview:")
    print(df.head())

    # Save the processed dataset for later use
    df.to_csv("processed_shipment_data.csv", index=False)
    print("\nProcessed dataset saved to 'processed_shipment_data.csv'")
    return df


if __name__ == '__main__':
    file_path = "shipment_data.xlsx"  # Replace with your actual file path
    df = prepare_data(file_path)