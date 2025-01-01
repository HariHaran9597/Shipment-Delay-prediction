# Shipment Delay Prediction

This project predicts whether a shipment will be delayed based on various factors such as distance, weather conditions, traffic conditions, and other shipment details. It uses machine learning models to classify shipments as delayed or on time.

---

## Project Overview

Shipping companies face significant challenges in ensuring on-time delivery. Delays can result in customer dissatisfaction and increased operational costs. This project aims to build a predictive model to identify shipments likely to be delayed, enabling proactive measures to minimize delays.

---

## Features

- **Data Preparation**:
  - Cleaned and processed raw shipment data.
  - Engineered features such as planned delivery duration, actual delivery delay, and date-based features.
  - Normalized and encoded data for machine learning.

- **Model Development**:
  - Logistic Regression as a baseline model.
  - Random Forest for handling non-linear feature interactions.
  - Hyperparameter tuning for improved model performance.

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score.
  - Confusion matrix visualization for detailed model evaluation.

---

## Dataset

The dataset consists of 20,000 rows with the following columns:

| Column Name                | Description                                      |
|----------------------------|--------------------------------------------------|
| `Shipment ID`              | Unique identifier for each shipment.            |
| `Origin`                   | Shipment origin city.                            |
| `Destination`              | Shipment destination city.                       |
| `Shipment Date`            | Date when the shipment was dispatched.           |
| `Planned Delivery Date`    | Planned delivery date of the shipment.           |
| `Actual Delivery Date`     | Actual delivery date of the shipment.            |
| `Vehicle Type`             | Type of vehicle used for the shipment.           |
| `Distance (km)`            | Distance between origin and destination (in km). |
| `Weather Conditions`       | Weather during the shipment period.              |
| `Traffic Conditions`       | Traffic conditions during the shipment.          |
| `Delayed`                  | Whether the shipment was delayed (`Yes`/`No`).   |

---

## Requirements

- Python 3.8 or above
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

Install the dependencies using:

```bash
pip install -r requirements.txt
