import pandas as pd
import numpy as np
from datetime import datetime
import car_data_prep

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import  ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, make_scorer
import pickle

df_original= pd.read_csv('dataset.csv')

df_p = car_data_prep.prepare_data(df_original,wb =False)


def remove_outliers(df, lower_quantile, upper_quantile, multiplier):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop('Price', errors='ignore')
    outlier_indices = set()
        
    for column_name in numeric_columns:
        Q1 = df[column_name].quantile(lower_quantile)
        Q3 = df[column_name].quantile(upper_quantile)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
            
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index
        outlier_indices.update(outliers)
        
    df_cleaned = df.drop(index=outlier_indices)
    return df_cleaned

df_p = remove_outliers(df_p, 0.23, 0.86, 2.2)

# Define features and target
X = df_p.drop(columns=['Price'])
y = df_p['Price']

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

# Recommended splits based on observations
splits = [
    ('Split 1', 0.15, 0.2353),
    ('Split 2', 0.10, 0.2222)
]
scalers = {
    'StandardScaler': StandardScaler(),
}

# Define the expanded parameter grid
param_distributions = {
    'regressor__alpha': np.logspace(-3.5, -1, 13),  
    'regressor__l1_ratio': np.linspace(0.1, 0.9, 14) }

# Define scoring metrics
scoring = {
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'neg_root_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False, squared=False),
    'r2': 'r2',
    'explained_variance': 'explained_variance'}

best_cv_rmse = np.inf
best_model_overall = None
best_params_overall = None
best_scaler_name = None
best_split_name = None

# Log transformation for numerical features
log_transformer = FunctionTransformer(np.log1p, validate=True)

for scaler_name, scaler in scalers.items():
    print(f"\nEvaluating scaler: {scaler_name}")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('poly_transform', PolynomialFeatures(degree=2, include_bias=False)),  # Apply polynomial transformation
        ('scaler', scaler)])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # Handle categorical features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet())])

    for split_name, test_size, val_size in splits:
        print(f"\nEvaluating {split_name} with test_size={test_size} and val_size={val_size}")

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

        random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=10, scoring=scoring, refit='neg_mean_squared_error', return_train_score=True, random_state=42)
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        y_val_pred = best_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Calculate standard deviation of cross-validation scores
        cv_rmse_scores = np.sqrt(-random_search.cv_results_['mean_test_neg_mean_squared_error'])
        val_std = np.std(cv_rmse_scores)
        print(f"Validation RMSE: {val_rmse}, Validation RMSE Std Dev: {val_std}")
        
        y_test_pred = best_model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_std = np.std(y_test - y_test_pred)
        print(f"Test RMSE:       {rmse_test},     Test RMSE Std Dev: {test_std}")

        print("Best parameters found: ", random_search.best_params_)

        if val_rmse < best_cv_rmse:
            best_cv_rmse = val_rmse
            best_model_overall = best_model
            best_params_overall = random_search.best_params_
            best_scaler_name = scaler_name
            best_split_name = split_name

print(f"\nBest model overall: Scaler: {best_scaler_name}, Split: {best_split_name}")
print(f"Best Cross-Validation RMSE: {best_cv_rmse}")
print(f"Best parameters: {best_params_overall}")

# Final model training with the best parameters
final_model = best_model_overall

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(final_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"\n10-Fold Cross-Validation RMSE scores: {cv_rmse_scores}")
print(f"Mean CV RMSE: {np.mean(cv_rmse_scores)}, Std Dev CV RMSE: {np.std(cv_rmse_scores)}")

# Fit the final model on the full training data
final_model.fit(X_train, y_train)
   
# Save the best model
pickle.dump(final_model, open("trained_model.pkl","wb"))