#import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
#_________________________________________________________________________________________

"""Features_importantes- Explanation
In this project, we aim to predict car prices accurately by selecting the most relevant features.
To manage the complexity and maintain clean, modular code, we have organized our feature selection process into several steps. This approach ensures a robust, reproducible, and maintainable workflow, facilitating efficient experimentation and model optimization.
Here's a step-by-step outline of the process:

Data Preparation:
Loading the Dataset: Begin by loading the dataset containing car features and prices.

Data Cleaning and Preprocessing: Handle missing values, transform categorical variables,
and apply necessary data transformations to prepare the dataset for analysis.

Feature Selection Techniques:

Univariate Selection: Use statistical tests to identify features with a strong relationship to car prices.
Recursive Feature Elimination (RFE): 
Iteratively remove features and build models to determine the most important ones.
Principal Component Analysis (PCA):
Reduce the dataset's dimensionality while retaining most of the variance to simplify the feature set.
Feature Importance from Models: 
Utilize machine learning models like ElasticNet to rank the importance of each feature.

Model Building and Evaluation:
Data Splitting: Divide the dataset into training and testing sets to validate the models.
Model Training: Train models using the selected features and assess their performance.
Model Tuning: Optimize model parameters to select the best-performing model.

Integration and Modularization:
Organizing Code into Functions: Define functions for each step of the process.
Importing Functions: Import these functions into the main Jupyter Notebook to maintain a clean and organized codebase.
Applying Functions: Execute the functions systematically within the main notebook to ensure a structured and efficient workflow.
By following this structured approach, we enhance the clarity and efficiency of our feature selection process, making it easier to debug, update, and extend in the future.
"""
#_________________________________________________________________________________________
"""This function updates a dictionary that keeps track of the frequency of each feature's selection. """
def update_feature_dict(feature_dict, features):
    for feature in features:
        feature_dict[feature] += 1
#_________________________________________________________________________________________
def feature_selection_pipeline(data, target_column='Price', degree=1, n_features_to_select=10):
"""This function builds and executes a feature selection pipeline for the input data.
    It performs the following steps:
    Data Preparation: Separates the target variable (y) from the feature set (X).
    Converts specified columns to categorical data types if needed.
    Preprocessing Pipelines: Creates separate pipelines for preprocessing numerical and categorical data.
    Feature Transformation: Applies the preprocessing pipelines to the data.
    Feature Selection Methods: Utilizes three different methods (ElasticNet, SelectKBest, and RFE)
    to select the top features.
    Feature Tracking: Updates a dictionary to track the frequency of each feature's selection.
    Final Feature Mapping: Maps the processed feature names back to their original names
    and aggregates their selection counts."""

    feature_dict = defaultdict(int)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Convert specific columns to categorical if needed
    if 'Total_Ownership' in X.columns:
        X['Total_Ownership'] = X['Total_Ownership'].astype('category')

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

    # Define preprocessing for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Process data
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    def get_top_features(model, X_processed, y, feature_names, n_features_to_select):
        model.fit(X_processed, y)
        coefs = np.abs(model.coef_) if hasattr(model, 'coef_') else model.feature_importances_
        indices = np.argsort(coefs)[-n_features_to_select:]
        return feature_names[indices], coefs[indices]

    # Apply ElasticNet model
    model = ElasticNet()
    selected_features, coefs = get_top_features(model, X_processed, y, feature_names, n_features_to_select)
    update_feature_dict(feature_dict, selected_features)

    # Apply SelectKBest
    bestfeatures = SelectKBest(score_func=f_regression, k=n_features_to_select)
    fit = bestfeatures.fit(X_processed, y)
    selected_features = feature_names[bestfeatures.get_support()]
    update_feature_dict(feature_dict, selected_features)

    # Apply RFE
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    fit = rfe.fit(X_processed, y)
    selected_features = feature_names[fit.support_]
    update_feature_dict(feature_dict, selected_features)

    # Map back to original feature names
    final_feature_dict = defaultdict(int)
    for feature in feature_dict.keys():
        if feature.startswith('num__'):
            original_name = feature.split('__', 1)[1]
            final_feature_dict[original_name] += feature_dict[feature]
        elif feature.startswith('cat__'):
            original_name = feature.split('__', 1)[1].split('_', 1)[0]
            final_feature_dict[original_name] += feature_dict[feature]
        else:
            final_feature_dict[feature] += feature_dict[feature]
    return final_feature_dict
#____________________________________________________________________________________________________


def get_top_features(data, target_column='Price', degree=1, n_features_to_select=10, top_X=5):
    """This function retrieves the top features for predicting car prices. It performs the following steps:
        Pipeline Execution: Calls the feature_selection_pipeline function to get a dictionary
        of selected features and their counts.
        Sorting and Selection: Sorts the features by their selection frequency in descending order.
        Top Features Extraction: Extracts the top top_X features based on their selection frequency
        and returns them."""
    feature_dict = feature_selection_pipeline(data, target_column=target_column, degree=degree, n_features_to_select=n_features_to_select)
    # Sort features by their weighted scores
    sorted_features = sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)
    top_features = [feature for feature, count in sorted_features[:top_X]]
    top_features = ['Total_Ownership' if 'Total_Ownership' in feature else feature for feature in top_features]
    return top_features
#____________________________________________________________________________________________________
