from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from car_data_prep import prepare_data
import traceback
import logging
from datetime import datetime

# Create Flask application object
app = Flask(__name__, static_folder='static', template_folder='templates')

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# Load the trained model
try:
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    model = None

# Home route, renders the main page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route, handles POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log received data
        data = request.form.to_dict()
        app.logger.debug(f"Received data: {data}")
       
        # Convert empty strings to None
        for key, value in data.items():
            if value == '':
                data[key] = None
        
        # Create DataFrame
        df = pd.DataFrame([data])
        app.logger.debug(f"DataFrame created: {df.to_dict()}")
        
        # Log raw data information
        app.logger.debug(f"Raw data shape: {df.shape}")
        app.logger.debug(f"Raw data columns: {df.columns.tolist()}")
        
        # Combine TestMonth and TestYear into a single Test field
        test_month = df['Input_TestMonth'].iloc[0]
        test_year = df['Input_TestYear'].iloc[0]
        if test_month and test_year:
            test_month = str(test_month).zfill(2)
            df['Test'] = f"01/{test_month}/{test_year}"
        else:
            df['Test'] = None
        df['Price'] = 0 
        
        # Rename columns to match prepare_data expectations
        column_mapping = {
            'Input_Manufacturer': 'manufactor',
            'Input_Model': 'model',
            'Input_ModelYear': 'Year',
            'Input_Hand': 'Hand',
            'Input_TransType': 'Gear',
            'Input_EngineType': 'Engine_type',
            'Input_EngineVolume': 'capacity_Engine',
            'Input_Km': 'Km',
            'Input_Color': 'Color',
            'Input_PrevHolder': 'Prev_ownership',
            'Input_CurrentHolder': 'Curr_ownership',
            'Input_Description': 'Description',
            'Input_PicNum': 'Pic_num',
            'Input_Area': 'Area',
            'Input_City': 'City',
            'Input_CreDate': 'Cre_date',
            'Input_RepubDate': 'Repub_date' 
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure all expected columns are present
        expected_columns = [
            'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 
            'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 
            'Pic_num', 'Cre_date', 'Repub_date', 'Description', 'Color', 'Km', 'Test']
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        
        # Convert data types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Hand'] = pd.to_numeric(df['Hand'], errors='coerce').astype('Int64')
        df['Pic_num'] = pd.to_numeric(df['Pic_num'], errors='coerce').astype('Int64')
        
        app.logger.debug(f"DataFrame before prepare_data: {df.to_dict()}")

        # Prepare data
        try:
            prepared_data = prepare_data(df, wb=True)
            df.drop(columns=['Price'], inplace=True)

            app.logger.debug(f"Prepared data: {prepared_data.to_dict()}")
            
            # Log prepared data information
            app.logger.debug(f"Prepared data shape: {prepared_data.shape}")
            app.logger.debug(f"Prepared data columns: {prepared_data.columns.tolist()}")
        except Exception as e:
            app.logger.error(f"Error in prepare_data: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": f"An error occurred during data preparation: {str(e)}"}), 501

        # Make prediction
        if model is None:
            return jsonify({"error": "Model not loaded"}), 502

        # Log model information
        app.logger.debug(f"Model expected features: {model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else 'Not available'}")
        
        try:
            prediction = model.predict(prepared_data)[0]
            if prediction <= 0:
                prediction = 0
            app.logger.debug(f"Raw prediction: {prediction}")
        except Exception as e:
            app.logger.error(f"Error in model prediction: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 503
        
        # Round prediction to nearest thousand
        rounded_prediction = round(prediction, -2)
        
        app.logger.debug(f"Rounded prediction: {rounded_prediction}")
        return f"{rounded_prediction:,.0f}"
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
