import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the PROCESSED dataset
file_path = "processed_shipment_data.csv"  # Corrected file path
df = pd.read_csv(file_path)

# Check class distribution
print("Class Distribution:")
print(df['Delayed'].value_counts())

# Plot class distribution
sns.countplot(x='Delayed', data=df)
plt.title('Class Distribution of Delayed Shipments')
plt.show()

print(df[['Shipment ID', 'Planned Delivery Duration', 'Actual Delivery Delay']].head())

# Plot histograms for numerical features
numerical_cols = ['Distance (km)', 'Planned Delivery Duration', 'Actual Delivery Delay']
df[numerical_cols].hist(bins=30, figsize=(10, 7))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Analyze the impact of Weather Conditions on Delays
sns.countplot(x='Weather Conditions', hue='Delayed', data=df)
plt.title('Impact of Weather Conditions on Delayed Shipments')
plt.show()

# Analyze the impact of Traffic Conditions on Delays
sns.countplot(x='Traffic Conditions', hue='Delayed', data=df)
plt.title('Impact of Traffic Conditions on Delayed Shipments')
plt.show()

# Plot correlation heatmap
# Option 1: Exclude 'Shipment ID' explicitly
#numerical_df = df.drop('Shipment ID', axis=1)
#correlation_matrix = numerical_df.corr()

# Option 2: Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=np.number).columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()