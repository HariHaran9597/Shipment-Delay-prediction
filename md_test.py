# Step 1: Import Necessary Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from dataprepartion import prepare_data




def train_and_evaluate_model(df):
    # Separate features and target
    X = df.drop(columns=['Delayed', 'Shipment ID'])
    y = df['Delayed']

    # Identify categorical and numerical columns
    categorical_cols = ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions']
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    time_delta_cols = ['Planned Delivery Duration', 'Actual Delivery Delay']

    # Step 3: Split the Dataset into Training and Testing Sets (using processed data)
    # Split the data, stratified to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

   # Create a column transformer for one-hot encoding categorical features and scaling numerical features
    preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
          ('num', StandardScaler(), numerical_cols),
          ('time_delta', 'passthrough', time_delta_cols)
      ],
       remainder='passthrough'
   )

    # Apply preprocessing to training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    # Apply preprocessing to test data
    X_test_processed = preprocessor.transform(X_test)
    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    # Convert data to DataFrame using correct feature names
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    print("Shape of training data: ", X_train_processed.shape)
    print("Shape of test data: ", X_test_processed.shape)
    print("\nFirst 5 rows of train data:\n", X_train_processed.head())

    # Step 4: Handle Class Imbalance (using Class weights)
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
    class_weight_dict = dict(zip([0, 1], class_weights))

    # Step 5: Model Selection - XGBoost with Class Weights or Resampling
    xgb_model = XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict.get(1, 1))  # Use scale_pos_weight for imbalance

    # Step 6: Model Selection - SVM with Class Weights or Resampling
    svm_model = SVC(random_state=42, class_weight='balanced')  # Use class_weight='balanced' for SVM to handle imbalance

    # Step 7: Hyperparameter Tuning with GridSearchCV
    # XGBoost Hyperparameter tuning
    xgb_param_grid = {
      'n_estimators': [100, 200],
      'max_depth': [3, 6],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.8, 1.0]
    }
    xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    xgb_grid_search.fit(X_train_processed, y_train) # Train using original train data

    # SVM Hyperparameter tuning
    svm_param_grid = {
       'C': [0.1, 1, 10],
       'kernel': ['linear', 'rbf'],
       'gamma': ['scale', 'auto']
    }
    svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    svm_grid_search.fit(X_train_processed, y_train) # Train using original train data

    # Step 8: Model Evaluation on the Test Set
    # XGBoost Evaluation
    xgb_best_model = xgb_grid_search.best_estimator_
    xgb_pred = xgb_best_model.predict(X_test_processed)

    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)

    # SVM Evaluation
    svm_best_model = svm_grid_search.best_estimator_
    svm_pred = svm_best_model.predict(X_test_processed)

    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred)
    svm_recall = recall_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)

    # Print Evaluation Results
    print("XGBoost Performance:")
    print(f"Accuracy: {xgb_accuracy:.4f}, Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f}, F1 Score: {xgb_f1:.4f}")
    print("\nSVM Performance:")
    print(f"Accuracy: {svm_accuracy:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, F1 Score: {svm_f1:.4f}")

    # Step 9: Visualize the Confusion Matrix
    def plot_confusion_matrix(y_true, y_pred, model_name):
      cm = confusion_matrix(y_true, y_pred)
      plt.figure(figsize=(6, 4))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Delayed', 'Delayed'], yticklabels=['Non-Delayed', 'Delayed'])
      plt.title(f'{model_name} - Confusion Matrix')
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.show()

    plot_confusion_matrix(y_test, xgb_pred, 'XGBoost')
    plot_confusion_matrix(y_test, svm_pred, 'SVM')

    # Step 10: Model Persistence
    # Save the best-performing model (XGBoost or SVM) using joblib
    best_model = xgb_best_model if xgb_f1 >= svm_f1 else svm_best_model
    joblib.dump(best_model, 'best_shipment_delay_model.pkl')

    print("Model saved as 'best_shipment_delay_model.pkl'")


# Main execution
if __name__ == '__main__':
    file_path = "shipment_data.xlsx"
    # Import and call the data processing function
    from dataprepartion import prepare_data
    df = prepare_data(file_path)
    train_and_evaluate_model(df)