# Import necessary libraries

import cleaning_funs #Python file attached
import column_prediction_models #Python file attached

import requests
import pandas as pd
import numpy as np
import datetime

import re
from datetime import datetime
from datetime import date, timedelta


def prepare_data(df,wb):
    """The prepare_data function preprocesses a dataset for modeling using utilities from the cleaning_funs module. It cleans text, 
        processes categories, converts data types, and generates new features such as car age. 
        The function also selects relevant columns and fills missing values, ensuring the data is ready for analysis."""

    # Ensure 'Description' column is of type string
    df['Description'] = df['Description'].astype(str)
    
    # Apply text cleaning functions
    df['Description'] = df['Description'].apply(cleaning_funs.clean_text_robust)
    df['model'] = df['model'].apply(cleaning_funs.clean_text_robust)
    
    # Clean 'capacity_Engine' column
    df['capacity_Engine'] = df['capacity_Engine'].replace(',', '', regex=True)
    
    # Apply custom functions to handle least frequent categories and conversions
    df = cleaning_funs.Color_least_frequent(df, 'Color', 'Processed_Colors',wb = wb)
    df = cleaning_funs.Engine_type_least_frequent(df, 'Engine_type')
    df = cleaning_funs.ownership_least_frequent(df, 'Curr_ownership', 'Processed_Curr_ownership')
    df = cleaning_funs.ownership_least_frequent(df, 'Prev_ownership', 'Processed_Prev_ownership')
    df = cleaning_funs.convert_to_date_and_calculate_days_and_check(df, 'Cre_date')
    df = cleaning_funs.convert_to_date_and_calculate_days_and_check(df, 'Repub_date')
    df = cleaning_funs.convert_Km(df, 'Km', 'Processed_Km')
    df = cleaning_funs.process_gear_column(df)
    df = cleaning_funs.process_manufactor_column(df)
    df = cleaning_funs.Processed_Test(df)
    df = cleaning_funs.convert_cap(df, 'capacity_Engine', 'Processed_capacity_Engine')
    df = cleaning_funs.update_description_length(df, 'Description', 'description_length')
    
    # Combine ownership columns
    df['Total_Ownership'] = df['Processed_Prev_ownership'].fillna('') + '_' + df['Processed_Curr_ownership']
    
    # Extract and process additional features
    df['dats_Test'] = df['Description'].apply(cleaning_funs.extract_test_dates_extended)
    df['Processed_Test'] = df.apply(cleaning_funs.apply_days_until_end_of_month, axis=1)
    df['Processed_model'] = df['model'].apply(cleaning_funs.translate_model)
    df = cleaning_funs.map_areas(df)

    # Convert columns to appropriate types
    if wb == False:
        df['Total_Ownership'] = df['Total_Ownership'].astype('category')
        df['Processed_model'] = df['Processed_model'].astype('category')
        df['Processed_manufactor'] = df['Processed_manufactor'].astype('category')
        df['Processed_Gear'] = df['Processed_Gear'].astype('category')
        df['Processed_Colors'] = df['Processed_Colors'].astype('category')
        df['Processed_Engine_type'] = df['Processed_Engine_type'].astype('category')
        df['Processed_Curr_ownership'] = df['Processed_Curr_ownership'].astype('category')
        df['Processed_Prev_ownership'] = df['Processed_Prev_ownership'].astype('category')
        df['Processed_Area'] = df['Processed_Area'].astype('category')

    df['Processed_Test'] = pd.to_numeric(df['Processed_Test'], errors='coerce').astype('Int64')
    
    # Create new features
    df['Car_Age'] = 2024 - df['Year']
   
    # Define the columns to copy
    columns_to_copy = [
        'Price', 'Car_Age', 'Hand', 'Pic_num', 'description_length', 
        'Processed_Colors','Processed_Engine_type', 'Processed_Area', 'Processed_Repub_date', 'Processed_capacity_Engine', 
        'Processed_Km', 'Processed_Gear','Processed_manufactor', 'Processed_Test','Processed_model',
        'Total_Ownership']
    
    # Select the relevant columns
    df_p = df[columns_to_copy]
    
    if wb == False:
       df_p = column_prediction_models.fill_missing_values_colors(df_p, 'Processed_Colors')
       df_p = column_prediction_models.fill_missing_values_ownership(df_p, 'Total_Ownership')
    
    df_p = df_p.drop_duplicates()

    # Return only dataset_3
    return df_p
