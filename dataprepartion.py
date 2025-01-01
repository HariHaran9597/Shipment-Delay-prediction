# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

# Load the dataset (for .xlsx files)
file_path = "shipment_data.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)  # Ensure the file path is correct

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (drop rows with missing values in essential columns)
essential_cols = ["Shipment Date", "Planned Delivery Date", "Actual Delivery Date", "Vehicle Type", "Delayed"]
df = df.dropna(subset=essential_cols)

# Convert date columns to datetime format
date_cols = ["Shipment Date", "Planned Delivery Date", "Actual Delivery Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')  # Handle invalid dates gracefully

# Drop rows with invalid dates
df = df.dropna(subset=date_cols)

# Feature Engineering
# Calculate Planned Delivery Duration (in days)
df['Planned Delivery Duration'] = (df['Planned Delivery Date'] - df['Shipment Date']).dt.days

# Calculate Actual Delivery Delay (in days)
df['Actual Delivery Delay'] = (df['Actual Delivery Date'] - df['Planned Delivery Date']).dt.days

# Drop rows with negative or unrealistic durations/delays
df = df[(df['Planned Delivery Duration'] >= 0) & (df['Actual Delivery Delay'] >= 0)]

# Extract additional date-related features
df['Shipment Month'] = df['Shipment Date'].dt.month
df['Shipment Day of Week'] = df['Shipment Date'].dt.dayofweek
df['Is Weekend'] = df['Shipment Day of Week'].apply(lambda x: 1 if x >= 5 else 0)

# Encode categorical features
categorical_cols = ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fill missing categorical values with a placeholder before encoding
    df[col] = df[col].fillna("Unknown")
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for potential decoding later

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = ['Distance (km)', 'Planned Delivery Duration', 'Actual Delivery Delay']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Clean and encode the target variable ('Delayed')
# Handle unexpected or invalid values
df['Delayed'] = df['Delayed'].str.strip()  # Remove leading/trailing spaces
df['Delayed'] = df['Delayed'].replace('', pd.NA)  # Replace empty strings with NaN
df['Delayed'] = df['Delayed'].where(df['Delayed'].isin(['Yes', 'No']), pd.NA)  # Keep only 'Yes' or 'No'
df = df.dropna(subset=['Delayed'])  # Drop rows with invalid or missing target values
df['Delayed'] = df['Delayed'].map({'Yes': 1, 'No': 0})  # Convert to binary

# Final dataset overview
print("\nProcessed Dataset Overview:")
print(df.head())

# Save the processed dataset for later use
processed_file_path = "processed_shipmentmodel_data.csv"
df.to_csv(processed_file_path, index=False)
print(f"\nProcessed dataset saved to '{processed_file_path}'")
