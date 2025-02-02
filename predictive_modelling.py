# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:57:09 2024

predictive_modeling.py
This script is designed to build and evaluate predictive models for customer churn and Customer Lifetime Value (CLV) using e-commerce data. It incorporates data preprocessing, 
feature engineering, hyperparameter tuning, and evaluation for both classification (churn) and regression (CLV) models. The script leverages machine learning algorithms such as 
XGBoost and RandomForest, and applies techniques like SMOTE to handle class imbalance. The outputs include predictions, feature importance plots, residual analysis, and model 
performance metrics, all of which are saved for further analysis and visualization in Power BI.

Key functionalities:

Data Preprocessing: Clean and preprocess the dataset, engineer additional features such as customer purchase frequency, time since last purchase, and product category preferences.
Hyperparameter Tuning: Tune hyperparameters for both classification and regression models using RandomizedSearchCV.
Churn Prediction: Train and evaluate a classifier to predict customer churn.
CLV Prediction: Train and evaluate a regression model to predict Customer Lifetime Value, with optional log transformation for skewed data.
Model Evaluation: Generate key metrics such as accuracy, mean squared error (MSE), and visualizations like residuals and feature importance.
Result Export: Save predictions, metrics, and plots for use in further analysis and reporting.

@author: Carlos
"""
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """
    Load the prepared dataset with metrics.
    """
    return pd.read_csv(filepath)


def prepare_features(df):
    """
    Prepare features for modeling. This includes selecting relevant columns,
    encoding categorical variables, and scaling features as necessary.
    """
    # Adding potential new features

    # 1. Customer Purchase Frequency
    purchase_frequency = df.groupby('Customer ID')['Purchase Date'].count().reset_index()
    purchase_frequency.columns = ['Customer ID', 'Purchase Frequency']
    df = pd.merge(df, purchase_frequency, on='Customer ID', how='left')

    # 2. Time Since Last Purchase
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    max_purchase_date = df.groupby('Customer ID')['Purchase Date'].max().reset_index()
    max_purchase_date.columns = ['Customer ID', 'Last Purchase Date']
    df = pd.merge(df, max_purchase_date, on='Customer ID', how='left')
    df['Days Since Last Purchase'] = (pd.to_datetime('today') - df['Last Purchase Date']).dt.days

    # 3. Product Category Preferences (One-hot encoded)
    category_pref = df.groupby(['Customer ID', 'Product Category']).size().unstack(fill_value=0)
    df = pd.merge(df, category_pref, on='Customer ID', how='left')

    # Adding Revenue per Purchase
    df['Revenue per Purchase'] = df['Total Purchase Amount'] / df['Quantity']
    
    # High Returns flag
    df['High Returns'] = df['Returns'] > 1

    # Define features for CLV Prediction (adding new features here)
    features = ['Customer Age', 'Gender', 'Total Purchase Amount', 'Returns', 'Revenue per Purchase', 
                'High Returns', 'Purchase Frequency', 'Days Since Last Purchase'] + list(category_pref.columns)

    # Separate features and target variables
    X = df[features]
    y_churn = df['Churn']
    y_clv = df['Customer Lifetime Value (CLV)']
    
    # Define preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Customer Age', 'Total Purchase Amount', 'Returns', 'Revenue per Purchase', 
                                       'Purchase Frequency', 'Days Since Last Purchase']),
            ('cat', OneHotEncoder(), ['Gender', 'High Returns'] + list(category_pref.columns))
        ])
    
    return X, y_churn, y_clv, preprocessor



def tune_hyperparameters(X, y, preprocessor, model_type='classifier'):
    """
    Tune hyperparameters for XGBoost models using RandomizedSearchCV within a Pipeline.
    Works for both classifier and regressor models.
    """
    # Define the pipeline
    if model_type == 'classifier':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_type == 'regressor':
        model = XGBRegressor(random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Define the hyperparameters to search
    param_grid = {
        'model__n_estimators': [50, 100, 200, 400],   # Number of boosting rounds
        'model__max_depth': [5, 10, 15, 20],          # Depth of each tree
        'model__learning_rate': [0.01, 0.1, 0.3],     # Learning rate or shrinkage
        'model__subsample': [0.7, 0.8, 1.0],          # Subsampling ratio
    }
    
    # Perform randomized search for more efficient tuning
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2, random_state=42, n_iter=20)
    random_search.fit(X, y)
    
    # Return the best model
    return random_search.best_estimator_


from sklearn.utils import resample

def churn_prediction(X, y, df, preprocessor):
    """
    Train a RandomForestClassifier to predict customer churn with hyperparameter tuning.
    """
    # Add a temporary index column to keep track of original indices
    X = X.copy()  # To avoid the SettingWithCopyWarning
    X['temp_index'] = df.index
    
    # Apply the preprocessing pipeline to X to transform categorical variables
    X_transformed = preprocessor.fit_transform(X.drop(columns=['temp_index']))
    
    # Convert sparse matrix to dense array before applying SMOTE
    if isinstance(X_transformed, scipy.sparse.csr_matrix):  # check if sparse
        X_transformed = X_transformed.toarray()
    
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_transformed, y)
    
    # Resample the original indices to match the SMOTE output
    resampled_indices = resample(X['temp_index'], n_samples=len(X_res), random_state=42)
    
    # Combine the resampled indices back into the DataFrame
    X_res_df = pd.DataFrame(X_res, columns=preprocessor.get_feature_names_out())
    X_res_df['temp_index'] = resampled_indices.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res_df, y_res, test_size=0.3, random_state=42)
    
    # Tune hyperparameters using the existing function
    clf = tune_hyperparameters(X_train.drop(columns=['temp_index']), y_train, preprocessor=None, model_type='classifier')
    
    # Train and evaluate the model
    clf.fit(X_train.drop(columns=['temp_index']), y_train)
    y_pred = clf.predict(X_test.drop(columns=['temp_index']))
    
    print("Churn Prediction Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Ensure the indices are correctly aligned
    churn_predictions = pd.DataFrame({
        'Customer ID': df.loc[X_test['temp_index'].values, 'Customer ID'].reset_index(drop=True),  # Use original indices
        'Actual Churn': y_test.reset_index(drop=True),
        'Predicted Churn': y_pred
    })
    
    # Check if the 'models' directory exists, and if not, create it
    if not os.path.exists('../models'):
        os.makedirs('../models')
        print("'models' directory created.")
    # Save the trained model
    joblib.dump(clf, '../models/tuned_churn_prediction_model.pkl')
    print("Tuned churn prediction model saved as 'models/tuned_churn_prediction_model.pkl'")
    
    return churn_predictions



def clv_prediction(X, y, df, preprocessor, apply_log=True, max_clv=160000):
    """
    Train a RandomForestRegressor to predict Customer Lifetime Value (CLV) with hyperparameter tuning.
    Optionally applies log transformation to target (CLV) to handle skewness.
    """
    # Add a temporary index column to keep track of original indices
    X = X.copy()  # To avoid the SettingWithCopyWarning
    X['temp_index'] = df.index
    
    # Apply the preprocessing pipeline to X to transform categorical variables
    X_transformed = preprocessor.fit_transform(X.drop(columns=['temp_index']))
    
    # Keep track of the original indices
    original_indices = X['temp_index'].values
    
    # Optionally apply log transformation to the target variable
    if apply_log:
        y = np.log1p(y)  # Use log1p to avoid log(0) issues
    
    # Split the data while preserving the original indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_transformed, y, original_indices, test_size=0.3, random_state=42)
    
    # Tune hyperparameters using the existing function
    reg = tune_hyperparameters(X_train, y_train, preprocessor=None, model_type='regressor')
    
    # Train and evaluate the model
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Cap predictions to avoid extremely large values after inverse transformation
    if apply_log:
        y_pred = np.clip(y_pred, a_min=None, a_max=np.log1p(max_clv))  # Cap predicted log values to max normal CLV
    
    # If log transformation was applied, revert back to original scale
    if apply_log:
        # Ensure that we do not encounter overflow
        try:
            y_test_exp = np.expm1(y_test)  # Revert log-transformed y_test
            y_pred_exp = np.expm1(y_pred)  # Revert log-transformed predictions
        except OverflowError as e:
            print(f"Overflow detected: {e}")
            y_test_exp = np.clip(np.expm1(y_test), a_min=0, a_max=max_clv)
            y_pred_exp = np.clip(np.expm1(y_pred), a_min=0, a_max=max_clv)
    
    # Calculate and print the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_exp, y_pred_exp)
    print(f"CLV Prediction Model Evaluation: MSE = {mse:.2f}")
    
    # Ensure the indices are correctly aligned
    clv_predictions = pd.DataFrame({
        'Customer ID': df.loc[test_indices, 'Customer ID'].reset_index(drop=True),  # Use original indices
        'Actual CLV': y_test_exp.reset_index(drop=True),  # Store actual CLV in original scale
        'Predicted CLV': y_pred_exp  # Store predicted CLV in original scale
    })
    
    
    # Check if the 'models' directory exists, and if not, create it
    if not os.path.exists('../models'):
        os.makedirs('../models')
        print("'models' directory created.")
    # Save the trained model
    joblib.dump(reg, '../models/tuned_clv_prediction_model.pkl')
    print("Tuned CLV prediction model saved as 'models/tuned_clv_prediction_model.pkl'")
    # Plot residuals
    plot_residuals(y_test_exp, y_pred_exp, title="Residuals for CLV Prediction", save_path='../results/clv_residuals.png')
    return clv_predictions

def plot_feature_importance(model, preprocessor, title="Top 5 Features (Churn prediction)", save_path=None):
    """
    Plot the feature importance of a RandomForest model and save the plot if a path is provided.
    """
    # Extract the model from the pipeline
    model = model.named_steps['model']
    
    # Extract the feature importances from the trained model
    importances = model.feature_importances_
    
    # Extract the feature names from the preprocessor (ColumnTransformer)
    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    numerical_features = ['Customer Age', 'Total Purchase Amount', 'Returns', 'Revenue per Purchase', 
                          'Purchase Frequency', 'Days Since Last Purchase']
    
    # Add more descriptive labels for product categories but not for 'Gender_Male'
    feature_names = list(numerical_features) + [
        f"{name} Purchases" if "Gender" not in name else name for name in onehot_features
    ]
    
    # Ensure feature names match the order of importances
    indices = importances.argsort()[::-1]
    
    # Select the top 5 most important features
    top_indices = indices[:5]
    top_feature_names = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Plot the top 5 features with larger text
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_feature_names, x=top_importances, palette="viridis")
    
    # Set title and axis labels with larger font size
    plt.title(title, fontsize=24)
    plt.xlabel("Importance", fontsize=18)
    plt.ylabel("Features", fontsize=18)
    
    # Increase the size of ticks on the axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Adjust layout to make space for longer feature names
    plt.subplots_adjust(left=0.3)  # Add more space on the left

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Ensure full plot is saved
        print(f"Feature importance plot saved as {save_path}")
    
    plt.show()

def plot_distribution(df, column, title="Distribution Plot", save_path=None):
    """
    Plot the distribution of a given column using Seaborn and save the plot if a path is provided.
    """
    # Drop duplicate Customer IDs to ensure each customer is represented only once
    df_unique = df.drop_duplicates(subset=['Customer ID'])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_unique[column], kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Distribution plot for {column} saved as {save_path}")
    
    plt.show()
    
def plot_residuals(y_test, y_pred, title="Residual Plot", save_path=None):
    """
    Plot the residuals (difference between actual and predicted values) to assess model performance.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle='--', color='red')
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Residual plot saved as {save_path}")
    
    plt.show()

# Load the prepared dataset
filepath = '../data/cleaned_ecommerce_customer_data_with_metrics.csv'
df = load_data(filepath)

# Prepare features and preprocessor for modeling
X, y_churn, y_clv, preprocessor = prepare_features(df)

# Train and evaluate Churn Prediction Model and get predictions
churn_predictions = churn_prediction(X, y_churn, df, preprocessor)


# Train and evaluate CLV Prediction Model and get predictions
clv_predictions = clv_prediction(X, y_clv, df, preprocessor)

# Combine all predictions into a single DataFrame
predictions_combined = pd.merge(churn_predictions, clv_predictions, on='Customer ID')

# Save the combined predictions to a CSV file
predictions_combined.to_csv('../results/predictions.csv', index=False)
print("All predictions saved to 'results/predictions.csv'.")

# Example of plotting and saving feature importance
clf_model = joblib.load('../models/tuned_churn_prediction_model.pkl')
plot_feature_importance(clf_model, preprocessor, save_path='../results/feature_importance_churn.png')

# Example of plotting and saving a distribution plot
plot_distribution(df, 'Customer Lifetime Value (CLV)', save_path='../results/clv_distribution.png')
