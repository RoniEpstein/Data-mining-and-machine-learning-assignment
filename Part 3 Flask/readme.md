
# Car Price Prediction Application

This project is a machine learning-based web application that predicts car prices based on various features. It uses a Flask backend with a trained ElasticNet regression model to provide price estimates for cars.

## Features

- Web interface for inputting car details
- Data preprocessing and cleaning
- Missing value prediction for model creation
- Machine learning model for price prediction
- Comprehensive error handling and logging

## Tech Stack

- Python 3.8+
- Flask 2.1.0
- pandas 1.3.3
- numpy 1.21.2
- scikit-learn 0.24.2
- Gunicorn 20.1.0 (for production deployment)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the trained model file `trained_model.pkl` in the project root directory.

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000` to use the application.

## Data Preparation

The application uses a custom data preparation module (`car_data_prep.py`) that performs the following steps:

- Cleans and processes raw data
- Handles missing values
- Encodes categorical variables
- Scales numerical features

## Explanation of Data Input

The web interface allows users to fill in all columns, and the model filters only the relevant columns for prediction. Some fields are designed for user convenience to enhance the user experience, while others are text inputs to demonstrate the robustness of the `prep_data` function. It can handle full data cleaning and processing as well as individual value handling. The design is playful and amusing to create an enjoyable user experience and make car price prediction lighter.

## Model Training

The model is trained using the following process:

1. Data splitting into train, validation, and test sets
2. Feature engineering including polynomial features
3. Hyperparameter tuning using RandomizedSearchCV
4. Model evaluation using multiple metrics (RMSE, MAE, R2, Explained Variance)
5. 10-fold cross-validation for robust performance estimation

## File Structure

- `app.py`: Main Flask application
- `car_data_prep.py`: Data preparation module
- `column_prediction_models.py`: Models for predicting missing values
- `train_model.py`: Script for training the price prediction model
- `requirements.txt`: List of Python dependencies
- `static/`: Directory for static files (CSS, JavaScript)
- `templates/`: Directory for HTML templates
- `trained_model.pkl`: Serialized trained model

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[Specify your license here, e.g., MIT License]

## Contact

[Odeya Hazani] - [Odeyah3@gmail.com]
[Roni Epstein] - [epstein@gmail.com]


Project Link: https://github.com/your-username/car-price-prediction
