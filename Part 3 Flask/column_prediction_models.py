import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, cross_val_predict, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


"""
column_prediction_models - Explanation
In this project, we are predicting and filling missing values in the 'color' and 
'total ownership' columns of our dataset as part of a machine learning assignment.
To manage the complexity and maintain clean, modular code, 
we have organized our modeling functions into a separate Python script.
Each function is designed to train a model for a specific column and use it to predict missing values.
By importing this script into our main Jupyter Notebook,
we can easily apply these modeling functions to the dataset. 
This approach not only helps in keeping our main notebook tidy and readable but also promotes code reusability and maintainability.
Here's a step-by-step outline of the process:
1. Separate Modeling Functions: Each column in the dataset ('color' and 'total ownership')
has a dedicated function that handles its prediction process.
These functions are defined in a separate Python script.
2. Importing the Script: The script containing the modeling functions is imported
into the main Jupyter Notebook using the import statement.
3. Applying the Functions: Once imported, these functions are applied to the respective columns 
in the dataset within the main notebook. 
This ensures that the missing values are predicted systematically and consistently.
By following this structured approach, we enhance the clarity and efficiency of our data preprocessing
and modeling workflow, making it easier to debug, update, and extend in the future.
"""

#________________________________________________________________
"""
remove_outliers - Function Explanation
This function removes outliers from numeric columns in a DataFrame
based on specified quantiles and a multiplier. 
Outliers are defined as values that fall outside the range determined by the interquartile range (IQR)
and the multiplier. 
The 'Price' column is excluded from this operation.
The function returns a cleaned DataFrame with the outliers removed.
"""
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

# _____________________________________________________________________________________________________________________________
"""
calculate_model_colors - Function Explanation
This function trains a machine learning model to predict the 'color' column in the dataset. 
It performs the following steps:
1. Removes outliers from the dataset.
2. Ensures column integrity by converting list-type entries to strings.
3. Selects features and the target column for training.
4. Handles missing values in the target column.
5. Converts the categorical target column to numerical values using LabelEncoder.
6. Creates separate pipelines for processing numerical and categorical data.
7. Combines these pipelines into a single preprocessor.
8. Defines and evaluates different models using cross-validation and hyperparameter tuning (RandomizedSearchCV).
9. Selects the best model based on precision score and refits it on the training+validation set.
10. Returns the best model and the label encoder for predicting missing 'color' values.
"""

def calculate_model_colors(df, target_column, seed=42, degree=2, splits=None):
    np.random.seed(seed)

    df = remove_outliers(df, 0.15, 0.85, 2.5) 

    # Ensure column integrity
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Feature and target selection
    X = df.drop(columns=['Price', target_column])
    y = df[target_column]

    # Handle missing values in y
    X = X[~y.isna()]
    y = y.dropna()

    # Convert categorical target column to numerical
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

    # Numerical data processing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('scaler', StandardScaler())])

    # Categorical data processing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine data processing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define cross-validation strategy with fewer folds
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    # Define the models and their parameter grids
    models = {
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        })
    }

    # Function to evaluate the model with RandomizedSearchCV
    def evaluate_model_with_search(model, param_grid, X_train, y_train):
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=kf, scoring='precision_weighted', n_jobs=-1, random_state=seed)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='precision_weighted', n_jobs=-1)
        return best_model, scores.mean()

    # Evaluate models on each split
    splits = splits or [
    #   ('Split 1', 0.30, 0.2857),  # 70% train_val, 30% test; 50% train, 20% val
    #    ('Split 2', 0.25, 0.20),    # 75% train_val, 25% test; 60% train, 15% val
        ('Split 3', 0.15, 0.2135)   # Adjusted split: ~85% train_val, 15% test; 78% train, 7% val
    ]

    best_overall_model = None
    best_overall_precision = float('-inf')

    for model_name, (model, param_grid) in models.items():
        # Base pipeline steps
        pipeline_steps = [('preprocessor', preprocessor)]
        pipeline_steps.append(('selector', SelectKBest(f_classif)))
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(steps=pipeline_steps)

        # Add parameter for k in SelectKBest
        param_grid.update({'selector__k': np.arange(1, X.shape[1] + 1)})

        for split_name, test_size, val_size in splits:
            # Split data into training+validation and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

            # Further split training+validation set into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=seed)

            best_model, mean_precision = evaluate_model_with_search(pipeline, param_grid, X_train, y_train)

            # Fit the best model on the training+validation set
            best_model.fit(X_train_val, y_train_val)

            # Predict on the test set
            y_pred = best_model.predict(X_test)
            test_precision = precision_score(y_test, y_pred, average='weighted')

            if test_precision > best_overall_precision:
                best_overall_model = best_model
                best_overall_precision = test_precision

    # Refit the entire pipeline on the best training+validation set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    best_overall_model.fit(X_train_val, y_train_val)

    return best_overall_model, label_encoder
 #___________________________________________________________________________________________________   
"""
fill_missing_values_colors - Function Explanation
This function predicts and fills missing values in the 'color' column using a trained machine learning model. 
The process involves:
1. Finding the best model and label encoder by calling the calculate_model_colors function.
2. Creating a copy of the DataFrame to avoid modifying the original data.
3. Identifying rows with missing values in the target column.
4. Using the trained model to predict missing values based on the same features used for training.
5. Converting predicted values back to their original categorical labels using the label encoder.
6. Filling the missing values in the copied DataFrame with the predicted values.
The function returns the DataFrame with the missing values filled.
"""

def fill_missing_values_colors(df, target_column, seed=42, degree=2, splits=None):
    # Step 1: Find the best model and label_encoder
    best_model, label_encoder = calculate_model_colors(df, target_column, seed, degree, splits)

    # Step 2: Fill missing values using the best model
    # Copy the DataFrame to avoid modifying the original one
    df_filled = df.copy()

    # Handle missing values in target column
    missing_index = df_filled[target_column].isna()

    # Check if there are missing values to predict
    if missing_index.sum() == 0:
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        return df_filled

    # Use the same features that were used for training the model
    X_full = df.drop(columns=['Price', target_column])
    X_missing = X_full.loc[missing_index]

    # Ensure the columns match the training data
    X_missing = X_missing[X_full.columns]

    # Predict missing values
    y_missing_pred = best_model.predict(X_missing)

    # Convert predictions back to original categories
    predicted_processed_colors = label_encoder.inverse_transform(y_missing_pred)

    # Fill missing values
    df_filled.loc[missing_index, target_column] = predicted_processed_colors

    return df_filled

# _____________________________________________________________________________________________________________________________
"""
calculate_model_ownership - Function Explanation
This function trains a machine learning model to predict the 'total ownership' column in the dataset. 
The process involves:
1. Removing outliers from the dataset.
2. Ensuring column integrity by converting list-type entries to strings.
3. Selecting features and the target column for training.
4. Handling missing values in the target column.
5. Converting the categorical target column to numerical values
using LabelEncoder and ensuring a continuous range of integers.
6. Creating separate pipelines for processing numerical and categorical data.
7. Combining these pipelines into a single preprocessor.
8. Defining and evaluating different models using cross-validation and hyperparameter tuning (RandomizedSearchCV).
9. Selecting the best model based on precision score and refitting it on the training+validation set.
10. Returning the best model and the label encoder for predicting missing 'total ownership' values.
"""

def calculate_model_ownership(df, target_column, seed=42, degree=2, splits=None):
    np.random.seed(seed)
    
    df = remove_outliers(df, 0.15, 0.85, 2.5) 

    # Ensure column integrity
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Feature and target selection
    X = df.drop(columns=['Price', target_column])
    y = df[target_column]

    # Handle missing values in y
    X = X[~y.isna()]
    y = y.dropna()

    # Convert categorical target column to numerical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Ensure continuous range of integers for y_encoded
    unique_classes = np.unique(y_encoded)
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    y_mapped = np.array([class_mapping[old_class] for old_class in y_encoded])

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

    # Numerical data processing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('scaler', StandardScaler())
    ])

    # Categorical data processing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine data processing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define cross-validation strategy with fewer folds
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    # Define the models and their parameter grids
    models = {
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        }),
    }

    # Function to evaluate the model with RandomizedSearchCV
    def evaluate_model_with_search(model, param_grid, X_train, y_train):
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=kf, scoring='precision_weighted', n_jobs=-1, random_state=seed)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='precision_weighted', n_jobs=-1)
        return best_model, scores.mean()

    # Evaluate models on each split
    splits = splits or [
        # ('Split 1', 0.30, 0.2857),  # 70% train_val, 30% test; 50% train, 20% val
        # ('Split 2', 0.25, 0.20),    # 75% train_val, 25% test; 60% train, 15% val
        ('Split 3', 0.15, 0.2135)   # Adjusted split: ~85% train_val, 15% test; 78% train, 7% val
    ]

    best_overall_model = None
    best_overall_precision = float('-inf')
    best_model_name = ""
    best_split_name = ""

    for model_name, (model, param_grid) in models.items():
        # Base pipeline steps
        pipeline_steps = [('preprocessor', preprocessor)]
        pipeline_steps.append(('selector', SelectKBest(f_classif)))
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(steps=pipeline_steps)

        # Add parameter for k in SelectKBest
        param_grid.update({'selector__k': np.arange(1, X.shape[1] + 1)})

        for split_name, test_size, val_size in splits:
            # Split data into training+validation and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_mapped, test_size=test_size, random_state=seed)

            # Further split training+validation set into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=seed)

            best_model, mean_precision = evaluate_model_with_search(pipeline, param_grid, X_train, y_train)

            # Fit the best model on the training+validation set
            best_model.fit(X_train_val, y_train_val)

            # Predict on the test set
            y_pred = best_model.predict(X_test)
            test_precision = precision_score(y_test, y_pred, average='weighted')

            if test_precision > best_overall_precision:
                best_overall_model = best_model
                best_overall_precision = test_precision
                best_model_name = model_name
                best_split_name = split_name

    # Refit the entire pipeline on the best training+validation set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_mapped, test_size=0.25, random_state=seed)
    best_overall_model.fit(X_train_val, y_train_val)

    return best_overall_model, label_encoder

#_________________________________________________________________________________________________
"""
fill_missing_values_ownership - Function Explanation
This function predicts and fills missing values in the 'total ownership' column using a trained machine learning model. 
The process involves:
1. Finding the best model and label encoder by calling the calculate_model_ownership function.
2. Creating a copy of the DataFrame to avoid modifying the original data.
3. Identifying rows with missing values in the target column.
4. Using the trained model to predict missing values based on the same features used for training.
5. Converting predicted values back to their original categorical labels using the label encoder.
6. Filling the missing values in the copied DataFrame with the predicted values.
The function returns the DataFrame with the missing values filled.
"""

def fill_missing_values_ownership(df, target_column, seed=42, degree=2, splits=None):
    # Step 1: Find the best model and label_encoder
    best_model, label_encoder = calculate_model_ownership(df, target_column, seed, degree, splits)

    # Step 2: Fill missing values using the best model
    # Copy the DataFrame to avoid modifying the original one
    df_filled = df.copy()

    # Handle missing values in target column
    missing_index = df_filled[target_column].isna()

    # Check if there are missing values to predict
    if missing_index.sum() == 0:
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        return df_filled

    # Use the same features that were used for training the model
    X_full = df.drop(columns=['Price', target_column])
    X_missing = X_full.loc[missing_index]

    # Ensure the columns match the training data
    X_missing = X_missing[X_full.columns]

    # Predict missing values
    y_missing_pred = best_model.predict(X_missing)

    # Convert predictions back to original categories
    predicted_processed_colors = label_encoder.inverse_transform(y_missing_pred)

    # Fill missing values
    df_filled.loc[missing_index, target_column] = predicted_processed_colors

    return df_filled
 #___________________________________________________________________________________________________   
