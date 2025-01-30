#Import necessary libraries

import requests
import pandas as pd
import datetime
import re
import numpy as np
from datetime import datetime
from datetime import date, timedelta
#from scipy.stats import chi2_contingency,f_oneway

# cleaninig_funs - Explanation
#
# In this project, we are cleaning and preprocessing various columns in our dataset as part of a machine learning assignment.
# To manage the complexity and maintain clean, modular code, we have organized our data cleaning functions into a separate Python script.
# Each function is designed to clean a specific column in the dataset.
#
# By importing this script into our main Jupyter Notebook, we can easily apply these cleaning functions to the dataset.
# This approach not only helps in keeping our main notebook tidy and readable but also promotes code reusability and maintainability.
#
# Here's a step-by-step outline of the process:
#
# Separate Cleaning Functions: Each column in the dataset has a dedicated function that handles its cleaning process.
# These functions are defined in a separate Python script.
#
# Importing the Script: The script containing the cleaning functions is imported into the main Jupyter Notebook using the import statement.
#
# Applying the Functions: Once imported, these functions are applied to the respective columns in the dataset within the main notebook.
# This ensures that the data is cleaned systematically and consistently.
#
# By following this structured approach, we enhance the clarity and efficiency of our data preprocessing workflow,
# making it easier to debug, update, and extend in the future.

#--------------------------------------------------------------------------------------------------------------------------------------------
""""" Function Explanation: extract_test_dates_extended
 The extract_test_dates_extended function extracts "Test Until" dates from a given description 
 string using various regex patterns to handle different date formats in Hebrew.
    Key Steps:
    Define Patterns:
    A list of regex patterns covers various date formats in Hebrew, including numeric dates and Hebrew month names.
    Hebrew Months Dictionary:
  A dictionary maps Hebrew month names to their numeric values.
  Pattern Matching:
  The function iterates over each pattern, searching for matches in the description.
  If a match is found, the function extracts the date components (day, month, year).
  Converts Hebrew month names to numeric values.
  Ensures the year is in four-digit format.
  Returns the extracted date as [year, month].
  Returns None if no pattern matches."""

def extract_test_dates_extended(description):
    """Extract 'Test Until' dates from the description based on various patterns."""
    
    patterns = [r"טסט עד ה (\d{1,2})\.(\d{2})",r"טסט (\d{2})/(\d{2})",r"טסט עד- (\d{1,2})\.(\d{4})",r"טסט-(\d{2})/(\d{4})",r"טסט עד התאריך (\d{1,2})/(\d{1,2})/(\d{4})",
        r"טסט ארוך עד ([א-ת]+) (\d{2})",r"טסט (\d{2})-(\d{2})-(\d{4})",r"טסט עד ([א-ת]+) (\d{4})",r"טסט ([א-ת]+) (\d{2})",r"טסט עד ([א-ת]+) (\d{2})",r"טסט בתחילת ([א-ת]+) (\d{4})",
        r"טסט ארוך - עד (\d{2})\.(\d{2})",r"טסט עד סוף חודש (\d{2})(\d{2})",r"טסט עד (\d)(\d{2})",r"טסט לשנה הבאה (\d)(\d{4})",r"טסט (\d)(\d{2})",r"טסט: (\d{2})(\d{2})",
        r"טסט עד סוף (\d{2})(\d{2})",r"טסט עד חודש ([א-ת]+) (\d{2})",r"טסט לעוד חצי שנה \((\d{2})/(\d{2})\)",r"טסט עד סוף ([א-ת]+) (\d{4})",r"טסט עד (\d{2})/(\d{4})מוביל",
        r"טסט היה ב([א-ת]+) (\d{4})",r"טסט עד חודש (\d{2})/(\d{4})",r"טסט מלא עד (\d{2})/(\d{2})/(\d{2})",r"טסט עד (\d{2})/(\d{2})",r"טסט עד (\d{2}) ([א-ת]+) (\d{4})",
        r"טסט עד סוף חודש (\d{2})/(\d{2})",r"טסט לשנה\(עד (\d{2})/(\d{2})\)",r"טסט עד סוף (\d{2})/(\d{2})",r"טסט עד ה- (\d{2})/(\d{2})",r"טסט (\d)/(\d{2})",
        r"טסט עד (\d{1,2})-(\d{1,2})-(\d{2,4})",r"טסט עד (\d{1,2})/(\d{1,2})/(\d{2,4})",r"טסט עד (\d{1,2})\.(\d{1,2})\.(\d{2,4})",r"טסט עד (\d{1,2})/(\d{4})",r"טסט עד (\d{2})/(\d{2})\!",r"טסט: עד ל (\d{1,2})/(\d{1,2})/(\d{2,4})",
        r"טסט עד ל (\d{1,2})\.(\d{4})",r"טסט עד (\d{2})/(\d{2})\S*",r"טסט לשנה הבאה (\d{1,2})/(\d{4})",r"טסט עד (\d{2})\.(\d{2})\S*",
    ]
    
    hebrew_months = {
        'ינואר': 1, 'פברואר': 2, 'מרץ': 3, 'אפריל': 4, 'מאי': 5, 'יוני': 6,
        'יולי': 7, 'אוגוסט': 8, 'ספטמבר': 9, 'אוקטובר': 10, 'נובמבר': 11, 'דצמבר': 12,
        'אוק': 10, 'אוג': 8  # Adding short forms
    }
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            if pattern in [
                r"טסט ארוך עד ([א-ת]+) (\d{2})",
                r"טסט עד ([א-ת]+) (\d{4})",
                r"טסט ([א-ת]+) (\d{2})",
                r"טסט עד ([א-ת]+) (\d{2})",
                r"טסט עד חודש ([א-ת]+) (\d{2})",
                r"טסט עד סוף ([א-ת]+) (\d{4})",
                r"טסט היה ב([א-ת]+) (\d{4})",
                r"טסט עד (\d{2}) ([א-ת]+) (\d{4})"
            ]:
                month_hebrew, year = match.groups()[-2:]
                month = hebrew_months.get(month_hebrew, None)
                if month is None:
                    continue  # skip if the month is not found in the dictionary
                year = int('20' + year) if len(year) == 2 and int(year) < 50 else int('19' + year) if len(year) == 2 else int(year)
                return [year, month]
            elif pattern in [
                r"טסט עד התאריך (\d{1,2})/(\d{1,2})/(\d{4})",
                r"טסט מלא עד (\d{2})/(\ד{2})/(\ד{2})"
            ]:
                day, month, year = match.groups()
                year = int('20' + year) if len(year) == 2 else int(year)
                return [year, int(month)]
            elif pattern in [
                r"טסט לעוד חצי שנה \((\ד{2})/(\ד{2})\)",r"טסט עד (\ד{2})/(\ד{4})מוביל",
                r"טסט עד חודש (\ד{2})/(\ד{4})",r"טסט עד סוף חודש (\ד{2})/(\ד{2})",r"טסט לשנה\(עד (\ד{2})/(\ד{2})\)",r"טסט עד סוף (\ד{2})/(\ד{2})",
                r"טסט עד ה- (\ד{2})/(\ד{2})",r"טסט עד (\ד{1,2})/(\ד{4})",r"טסט עד (\ד{2})/(\ד{2})\!",r"טסט: עד ל (\ד{1,2})/(\ד{1,2})/(\ד{2,4})", 
                r"טסט עד ל (\ד{1,2})\.(\ד{4})",r"טסט עד (\ד{2})/(\ד{2})\S*",r"טסט לשנה הבאה (\ד{1,2})/(\ד{4})",r"טסט עד (\ד{2})\.(\ד{2})\S*"
            ]:
                month, year = match.groups()
                year = int('20' + year) if len(year) == 2 else int(year)
                return [year, int(month)]
            elif pattern == r"טסט עד סוף חודש (\ד{2})(\ד{2})":
                month, year = match.groups()
                year = int('20' + year)
                return [year, int(month)]
            elif pattern in [
                r"טסט עד (\ד{1,2})-(\ד{1,2})-(\ד{2,4})",r"טסט עד (\ד{1,2})/(\ד{1,2})/(\ד{2,4})",r"טסט עד (\ד{1,2})\.(\ד{1,2})\.(\ד{2,4})"]:
                day, month, year = match.groups()
                year = int('20' + year) if len(year) == 2 else int(year)
                return [year, int(month)]
            else:
                parts = match.groups()
                if len(parts) == 2:
                    month, year = parts
                    month = hebrew_months.get(month, month)  # try to convert month if it's a Hebrew name
                    year = int('20' + year) if len(year) == 2 else int(year)
                    return [year, int(month)]
                elif len(parts) == 3:
                    day, month, year = parts
                    month = hebrew_months.get(month, month)  # try to convert month if it's a Hebrew name
                    year = int(year)
                    return [year, int(month)]
    return None
#--------------------------------------------------------------------------------------------------------------------------------------------

model_translation = {
    'i35': 'i35', 'מיקרה': 'Micra', 'סוויפט': 'Swift', 'אוריס': 'Auris', 'פיקנטו': 'Picanto',
    'A1': 'A1', 'אימפרזה': 'Impreza', 'ASX': 'ASX', '220': '220', '525': '525', 'מוקה': 'Mokka',
    'פורטה': 'Forte', 'Q3': 'Q3', 'סיוויק סדאן': 'Civic Sedan', 'SX4 קרוסאובר': 'SX4 Crossover',
    'קורולה': 'Corolla', 'גולף': 'Golf', 'פאסאט': 'Passat', 'ספארק': 'Spark', '3': '3',
    'נוט': 'Note', 'סול': 'Soul', 'V40 CC': 'V40 CC', 'לנסר ספורטבק': 'Lancer Sportback',
    'i10': 'i10', 'A3': 'A3', 'פאביה': 'Fabia', 'אוקטביה': 'Octavia', 'CIVIC': 'Civic',
    'איוניק': 'Ioniq', 'סונטה': 'Sonata', 'i30': 'i30', 'C-HR': 'C-HR', 'מאליבו': 'Malibu',
    'ריו': 'Rio', 'פוקוס': 'Focus', 'X1': 'X1', 'אוואו': 'Aveo', 'סיוויק': 'Civic',
    'E-Class': 'E-Class', 'S7': 'S7', 'אפלנדר': 'Outlander', 'SVX': 'SVX', 'איגניס': 'Ignis',
    'ספייס סטאר': 'Space Star', 'IS300h': 'IS300h', 'C4': 'C4', '2008': '2008', 'סטוניק': 'Stonic',
    'פולו': 'Polo', 'S60': 'S60', 'RS5': 'RS5', "Jazz Hybrid": 'Jazz Hybrid', 'SX4': 'SX4',
    'גטה': 'Jetta', 'A4': 'A4', 'אס-מקס': 'S-Max', 'נירו': 'Niro', 'אינסייט': 'Insight',
    'קליאו': 'Clio', 'All Road': 'All Road', 'פאסאט CC': 'Passat CC', 'S-Class': 'S-Class',
    'CADDY COMBI': 'Caddy Combi', 'אסטרה': 'Astra', 'XV': 'XV', 'A5': 'A5', '316': '316', 'C3': 'C3',
    'סדרה 5': '5 Series', 'אקורד': 'Accord', 'i25': 'i25', 'C1': 'C1', 'יאריס': 'Yaris',
    'IS250': 'IS250', 'V40': 'V40', 'סדרה 1': '1 Series', 'סראטו': 'Cerato', '5': '5',
    'דור 4': '4th Gen', 'קורבט': 'Corvette', "אטראז'": 'Attrage', 'i20': 'i20', '200': '200', 'B4': 'B4',
    '308': '308', "האצ'בק": 'Hatchback', 'מוקה X': 'Mokka X', 'זאפירה': 'Zafira', 'אינסיגניה': 'Insignia',
    '6': '6', 'CT200H': 'CT200H', 'אורלנדו': 'Orlando', 'אלתימה': 'Altima', 'אלטו': 'Alto',
    'קרוסאובר': 'Crossover', '108': '108', 'DS3': 'DS3', 'פריוס': 'Prius', 'שירוקו': 'Scirocco',
    "JUKE": 'Juke', 'XCEED': 'XCEED', 'אסטייט': 'Estate', 'ספייס': 'Space', 'פלואנס': 'Fluence',
    'SLK': 'SLK', 'אלנטרה': 'Elantra', 'S7': 'S7', 'בלנו': 'Baleno', 'טראקס': 'Trax',
    'FR-V': 'FR-V', 'סנטרה': 'Sentra', 'סיריון': 'Sirion', "Jazz": 'Jazz', 'גרנדיס': 'Grandis',
    'פרייד': 'Pride', 'סלבריטי': 'Celebrity', 'טריוס': 'Terios', 'חשמלי': 'Electric', '120i': '120i',
    'קאונטרימן': 'Countryman', '159': '159', 'MITO': 'Mito', 'אקווינוקס': 'Equinox', '208': '208',
    'A6': 'A6', 'חיפושית': 'Beetle', 'ראפיד': 'Rapid', 'B3': 'B3', 'ורסו': 'Verso', 'קורסה': 'Corsa',
    'ייטי': 'Yeti', 'אדם': 'Adam', "ג'ולייטה": 'Giulietta', 'סדן': 'Sedan', 'אאוטלנדר': 'Outlander',
    'מוסטנג': 'Mustang', 'GS300': 'GS300', '508': '508', 'RC': 'RC', 'ולוסטר': 'Veloster', 'קופר': 'Cooper',
    'CLK': 'CLK', 'EV': 'EV', 'קרוז': 'Cruze', 'ספלאש': 'Splash', 'גולף פלוס': 'Golf Plus', 'סופרב': 'Superb',
    'אוקטביה קומבי': 'Octavia Combi', 'PHEV': 'PHEV', 'S80': 'S80', 'לאונה': 'Leon', 'אקליפס': 'Eclipse',
    'סוניק': 'Sonic', '307CC': '307CC', 'אאוטבק': 'Outback', 'סדרה 3': '3 Series', 'טריוס': 'Terios',
    'אודסיי': 'Odyssey', 'קרניבל': 'Carnival', 'RS': 'RS', 'סלריו': 'Celerio', 'AX': 'AX', 'פיאסטה': 'Fiesta',
    'גלאקסי': 'Galaxy', 'פרימרה': 'Primera', 'וויאגר': 'Voyager', 'M1': 'M1', 'GTI': 'GTI',
    'GT3000': 'GT3000', 'R8': 'R8', 'סטיישן': 'Station', '300C': '300C', 'אימפלה': 'Impala', 'ספיה': 'Sephia',
    'קורסה החדשה': 'New Corsa', 'C-Class': 'C-Class', 'INSIGHT': 'Insight', "טרג'ט": 'Target', 'CX': 'CX',
    'קאמרי': 'Camry', '5008': '5008', 'אונסיס': 'Onyx', 'סיד': 'Ceed', '530': '530', 'i40': 'i40',
    '301': '301', 'C5': 'C5', '320': '320', 'Taxi': 'Taxi', '318': '318', 'C-CLASS': 'C-Class',
    'מריבה': 'Meriva', 'קופה': 'Coupe', 'הייבריד': 'Hybrid', 'קורסיקה': 'Corsica', 'קונקט': 'Connect',
    'RCZ': 'RCZ', 'קווסט': 'Quest', 'S5': 'S5', 'S3': 'S3', 'קאנטרימן': 'Countryman', '25': '25','טוראן':'Tiguan'}

def translate_model(model_name):
    """" Function Explanation: translate_model
         The translate_model function translates a given car model name from Hebrew to its corresponding
         English name using a predefined dictionary of translations."""
    # Check for the full model name in the dictionary
    if model_name in model_translation:
        return model_translation[model_name]
    
    # If not found, split the model name into words and check each word
    words = model_name.split()
    for word in words:
        if word in model_translation:
            return model_translation[word]
    
    return None
   
#--------------------------------------------------------------------------------------------------------------------------------------------

# Function to calculate days until the end of the month
def days_until_end_of_month(year, month):
    """ Function Explanation: days_until_end_of_month
        The days_until_end_of_month function calculates the number of days
        from today until the end of a specified month and year."""
    # Today's date
    today = date(2024, 6, 20)
    
    # First day of the given month and year
    first_of_month = date(year, month, 1)
    
    # Last day of the given month
    if month == 12:
        last_of_month = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_of_month = date(year, month + 1, 1) - timedelta(days=1)

    # Calculating days from today to the end of the month
    if today > last_of_month:
        # If today is past the end of the month, return 0 days
        return 0
    elif today < first_of_month:
        # If today is before the month starts, calculate from today to the last day of the month
        return (last_of_month - today).days 
    else:
        # Normal case within the month, calculate from today to the last day of the month
        return (last_of_month - today).days
      
#--------------------------------------------------------------------------------------------------------------------------------------------

def convert_to_year_month(date_str):
    """Function Explanation: convert_to_year_month
        The convert_to_year_month function converts a date string into a year and month tuple,
        handling various date formats."""
    
    try:
        # Try to parse date in the format 'dd/mm/yyyy'
        if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
            parsed_date = pd.to_datetime(date_str, format='%d/%m/%Y')
            if parsed_date.day == 1:  # If the day is the 1st, get the last day of the previous month
                parsed_date -= pd.DateOffset(days=1)
            return parsed_date.year, parsed_date.month
        
        # Try to parse date in the format 'mmm-yy'
        elif re.match(r'[A-Za-z]{3}-\d{2}', date_str):
            parsed_date = pd.to_datetime(date_str, format='%b-%y')
            return parsed_date.year, parsed_date.month
        
        # Handle any other date format that pandas can parse
        else:
            parsed_date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(parsed_date):
                if parsed_date.day == 1:  # If the day is the 1st, get the last day of the previous month
                    parsed_date -= pd.DateOffset(days=1)
                return parsed_date.year, parsed_date.month
    except Exception:
        return None
#--------------------------------------------------------------------------------------------------------------------------------------------
"""Function Explanation: Processed_Test
   The Processed_Test function updates a DataFrame's Processed_Test column with the number of days
   until the end of the month based on the Test column values.
   Converts date strings to year and month, calculates days until the month's end"""
def Processed_Test(df):
    def process_test_column(value):
        if isinstance(value, str) and not value.isdigit():
            result = convert_to_year_month(value)
            if result is not None:
                year, month = result
                return days_until_end_of_month(year, month)
            else:
                return None
            return value
    
    df['Processed_Test'] = df['Test']
    df['Processed_Test'] = df['Processed_Test'].apply(lambda x: 0 if isinstance(x, str) and x.lstrip('-').isdigit() and int(x) < 0 else x) #Converts negative string values to 0.
    df['Processed_Test'] = df['Processed_Test'].apply(process_test_column)
    df['Processed_Test'] = df['Processed_Test'].replace('None', pd.NA, regex=True)  #Replaces 'None' with pandas NA.   
    df['Processed_Test'] = pd.to_numeric(df['Processed_Test'], errors='coerce').astype('Int64') 
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def clean_text_robust(text):
    """Function Explanation: clean_text_robust
       The clean_text_robust function cleans a text string by removing unwanted characters,
       HTML tags, and other elements."""
    # Remove the first and last characters
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
        
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    # Remove HTML tags and entities
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&.*?;', '', text)
    # Remove specific problematic characters and backslashes
    text = text.replace("\\r\\n", "").replace("\\n", "").replace("\\r", "")
    text = text.replace("\r\n", "").replace("\n", "").replace("\r", "")
    text = text.replace("br/", "").replace("\\", "")
    return text.strip()
#--------------------------------------------------------------------------------------------------------------------------------------------


def process_manufactor_column(df):
    """Function Explanation: process_manufactor_column
        The process_manufactor_column function translates the manufactor column in a DataFrame to Hebrew
        using a predefined dictionary"""
    
    translation_dict = {'Lexsus': 'לקסוס','Ford': 'פורד','Volkswagen': 'פולקסווגן','Chevrolet': 'שברולט','Hyundai': 'יונדאי','Nissan': 'ניסאן','Suzuki': 'סוזוקי',
                    'Toyota': 'טויוטה','Kia': 'קיה', 'Audi': 'אאודי', 'Subaru': 'סובארו','Mitsubishi': 'מיצובישי', 'Mercedes': 'מרצדס', 'BMW': 'ב.מ.וו','Opel': 'אופל',
                    'Honda': 'הונדה', 'Mazda': 'מאזדה','Volvo': 'וולוו','Skoda': 'סקודה','Chrysler': 'קרייזלר','Citroen': 'סיטרואן','Peugeot': "פיג'ו",'Renault': 'רנו',
                    'Daihatsu': 'דייהטסו','Mini': 'מיני','Alfa Romeo': 'אלפא רומיאו','None': None}

    def translate_manufactor(value):
        return translation_dict.get(value, value)

    df['Processed_manufactor'] = df['manufactor'].apply(translate_manufactor)
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def Color_least_frequent(df, column_name, new_column_name, top_n=8):
    """Function Explanation: Color_least_frequent
        The Color_least_frequent function processes a DataFrame to replace the values in a specified color column
        with more general categories and groups less frequent colors into an "else" category."""

    # Replace 'None' values with 'else' and preserve NaN values
    df[new_column_name] = df[column_name].replace('None',  pd.NA)
    
    # Define a function to merge similar categories
    def merge_similar_categories(value):
        if isinstance(value, str):
            if 'לבן' in value or 'שמפניה' in value:
                return 'לבן'
            elif 'כחול' in value:
                return 'כחול'
            elif 'אפור' in value:
                return 'אפור'
            elif 'ירוק' in value:
                return 'ירוק'
            elif 'אדום' in value:
                return 'אדום'
            elif 'כסוף' in value or 'כסף' in value:
                return 'כסף'
            elif 'זהב' in value:
                return 'זהב '
            else:
                return value
        return value

    # Apply the merging function to the new column
    df['Merged'] = df[new_column_name].apply(merge_similar_categories)

    # Count the frequency of each string in the merged column
    value_counts = df['Merged'].value_counts()

    # Get the top N most frequent strings
    top_values = value_counts.nlargest(top_n).index

    # Replace all other strings with "else" in the new column
    df[new_column_name] = df['Merged'].apply(lambda x: x if x in top_values else 'else')

    # Preserve the original NaN values in the new column
    df.loc[df[column_name].isnull(), new_column_name] = pd.NA

    # Drop the temporary 'Merged' column
    df.drop(columns=['Merged'], inplace=True)

    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def Engine_type_least_frequent(df, column_name): 
    """Function Explanation: Engine_type_least_frequent
       The Engine_type_least_frequent function processes a DataFrame to merge similar engine type
       categories into more general categories."""
    # Define a function to merge similar categories
    def merge_similar_categories(value):
        if isinstance(value, str):
            if 'היבריד' in value:
                return 'היברידי'
            elif 'דיזל' in value:
                return 'דיזל'
        return value
    
    # Apply the merging function to the new column
    df['Processed_Engine_type'] = df[column_name].apply(merge_similar_categories)
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def process_gear_column(df):
    """Function Explanation: process_gear_column
    The process_gear_column function processes a DataFrame's Gear column to standardize gear type values."""

    # Define a function to merge similar categories
    def process_gear(value):
        if pd.isna(value) or value == 'לא מוגדר':
            return None
        elif value == 'אוטומט':
            return 'אוטומטית'
        else:
            return value
        
    # Apply the merging function to the new column
    df['Processed_Gear'] = df['Gear'].apply(process_gear)
    return df
   
#--------------------------------------------------------------------------------------------------------------------------------------------

def replace_least_frequent(df, column_name, top_n=6):
    """Function Explanation: replace_least_frequent
        The replace_least_frequent function processes a DataFrame 
        to replace less frequent values in a specified column with "else",
        keeping only the top N most frequent values"""
    
    # Count the frequency of each string in the column
    value_counts = df[column_name].value_counts()
    
    # Get the top N most frequent strings
    top_values = value_counts.nlargest(top_n).index
    
    # Replace all other strings with "else"
    df[f'least_frequent{column_name}'] = df[column_name].apply(lambda x: x if x in top_values else 'else')
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def convert_Km(df, column_name, new_column_name):
    """Function Explanation: convert_Km
        The convert_Km function processes a DataFrame to clean and convert the values
        in a specified column representing kilometers. 
        It handles special cases like zeros, commas, and 'None' values."""
    df[new_column_name] = df[column_name]
    
    # Replace '0' with pd.NA based on the 'Year' column condition
    df.loc[df['Year'] < 2024, new_column_name] = df[new_column_name].replace('0', pd.NA)
    
    # Remove commas
    df[new_column_name] = df[new_column_name].str.replace(',', '', regex=True)
    
    # Replace 'None' with pd.NA
    df[new_column_name] = df[new_column_name].replace('None', pd.NA, regex=True)
    
    # Convert to numeric and handle any conversion issues
    df[new_column_name] = pd.to_numeric(df[new_column_name], errors='coerce').astype('Int64')
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def convert_to_date_and_calculate_days_and_check(df, column_name):
    """Function Explanation: convert_to_date_and_calculate_days_and_check
        The convert_to_date_and_calculate_days_and_check function 
        processes a DataFrame column to parse date strings, calculate the number of days from a specified date, 
        and check if the date is after a given reference date."""
    
    date_of_interest = pd.Timestamp(date(2024, 6, 20))
    new_column_name = f'Processed_{column_name}'

    def try_parse_date(value):
        try:
            # Try parsing the value as a date
            parsed_date = pd.to_datetime(value, format='%d/%m/%Y', errors='coerce')
            if not pd.isnull(parsed_date):
                # Check if the date is after date_of_interest
                if parsed_date > date_of_interest:
                    return np.nan
                return (date_of_interest - parsed_date).days
            else:
                return np.nan
        except (ValueError, TypeError):
            return np.nan

    # Apply the function to the column and create a new column with the results
    df[new_column_name] = df[column_name].apply(try_parse_date)

    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def process_accident_column(df, description_column):
    """Function Explanation: process_accident_column
       The process_accident_column function processes a DataFrame to identify and flag descriptions that mention accidents"""
    def extract_surrounding_accident(text):
        pattern = r'.{0,15}תאונ.{0,15}'
        matches = re.findall(pattern, text)
        return ' '.join(matches) if matches else ''

    def assign_binary_flag(text):
        return 1 if text else 0

    df['Processed_accident_temp'] = df[description_column].apply(extract_surrounding_accident)

    df['Processed_accident'] = df['Processed_accident_temp'].apply(assign_binary_flag)

    df.drop(columns=['Processed_accident_temp'], inplace=True)

    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def convert_cap(df, column_name, new_column_name):
    """Function Explanation: convert_cap The convert_cap function processes a DataFrame column to clean and convert its values, 
        ensuring they fall within a specified range."""
    df[new_column_name] = df[column_name]
    
    # Remove commas
    df[new_column_name] = df[new_column_name].str.replace(',', '', regex=True)
    
    # Replace 'None' with pd.NA
    df[new_column_name] = df[new_column_name].replace('None', pd.NA, regex=True)
    
    # Convert to numeric and handle any conversion issues
    df[new_column_name] = pd.to_numeric(df[new_column_name], errors='coerce').astype('Int64')
    
    # Replace values less than 600 or greater than 7000 with pd.NA
    df.loc[(df[new_column_name] < 600) | (df[new_column_name] > 7000), new_column_name] = pd.NA
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def update_description_length(df, description_column, length_column):
    """Function Explanation: update_description_length
        The update_description_length function calculates the length of descriptions in a specified column
        and updates another column with these lengths, setting lengths to 0 for certain conditions."""

    df[length_column] = df[description_column].apply(lambda x: len(str(x)))
    df.loc[df[description_column].isin([np.nan, 'None', 'No description', 'no description']), length_column] = 0
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def fill_supply_score(df):
    """Function Explanation: fill_supply_score
        The fill_supply_score function fills missing values in the Supply_score 
        column of a DataFrame based on unique combinations of Year, Processed_manufactor, and Processed_model."""

    # Create a dictionary to store unique combinations and their corresponding Supply_score values
    supply_score_dict = {}
    
    # Populate the dictionary with unique combinations and their corresponding Supply_score values
    for index, row in df.iterrows():
        key = (row['Year'], row['Processed_manufactor'], row['Processed_model'])
        if pd.notna(row['Supply_score']):
            supply_score_dict[key] = row['Supply_score']
    
    # Define a function to fill Supply_score for each row
    def apply_supply_score(row):
        key = (row['Year'], row['Processed_manufactor'], row['Processed_model'])
        return supply_score_dict.get(key, row['Supply_score'])
    
    # Apply the function to the DataFrame
    df['Processed_Supply_score'] = df.apply(lambda row: apply_supply_score(row), axis=1)
    
    return df

#--------------------------------------------------------------------------------------------------------------------------------------------

# Function to apply the calculation conditionally
def apply_days_until_end_of_month(row):
    """Function Explanation: apply_days_until_end_of_month
        The apply_days_until_end_of_month function conditionally calculates
        the number of days until the end of the month based on a dats_Test value if Processed_Test is missing."""

    dats_Test = row['dats_Test']
    if pd.isna(row['Processed_Test']) and dats_Test is not None and all(x is not None for x in dats_Test):
        year, month = dats_Test
        if 1 <= month <= 12:
            return days_until_end_of_month(year, month)
        else:
            return pd.NA  # or handle this case appropriately
    return row['Processed_Test']
#--------------------------------------------------------------------------------------------------------------------------------------------

def map_areas(df, area_column='Area'):
    """Function Explanation: map_areas
       The map_areas function standardizes area names in a DataFrame
        by mapping them to predefined categories using a dictionary."""

    # Define the mapping dictionary
    area_mapping = {
        'מושבים': 'מושבים','ירושלים': 'ירושלים','באר שבע': 'באר שבע','חיפה': 'חיפה','קריה': 'קריות','קריות': 'קריות','ראשל"צ': 'ראשון לציון'
        ,'ראשון לציון': 'ראשון לציון','חולון': 'חולון ובת ים','בת ים': 'חולון ובת ים','אשדוד': 'אשדוד ואשקלון', 'אשקלון': 'אשדוד ואשקלון', 'גליל': 'גליל, עמק וכרמיאל',
        'עמק': 'גליל, עמק וכרמיאל', 'כרמיאל': 'גליל, עמק וכרמיאל','רמת גן': 'רמת גן, גבעתיים ופתח תקוה','גבעתיים': 'רמת גן, גבעתיים ופתח תקוה','פתח תקוה': 'רמת גן, גבעתיים ופתח תקוה'
        , 'פתח תקווה': 'רמת גן, גבעתיים ופתח תקוה', 'עכו': 'עכו ונהריה','נהריה': 'עכו ונהריה','נהרייה': 'עכו ונהריה','תל אביב': 'תל אביב','נס ציונה': 'נס ציונה, רחובות ורמלה','רחובות': 'נס ציונה, רחובות ורמלה',
        'רמלה': 'נס ציונה, רחובות ורמלה','נתניה': 'נתניה ורעננה'
        ,'רעננה': 'נתניה ורעננה','מודיעין': 'מודיעין','שרון': 'השרון והרצליה', 'השרון': 'השרון והרצליה','הרצליה': 'השרון והרצליה','ראש העין': 'ראש העין','בית שמש': 'בית שמש','אונו': 'אונו','חדרה': 'חדרה ועמק חפר'
        ,'עמק חפר': 'חדרה ועמק חפר','טבריה': 'טבריה','גדרה': 'גדרה ויבנה','יבנה': 'גדרה ויבנה','קיסריה': 'קיסריה, פרדס חנה, כרכור, זכרון ובנימינה','פרדס חנה': 'קיסריה, פרדס חנה, כרכור, זכרון ובנימינה',
        'כרכור': 'קיסריה, פרדס חנה, כרכור, זכרון ובנימינה','זכרון': 'קיסריה, פרדס חנה, כרכור, זכרון ובנימינה','בנימינה': 'קיסריה, פרדס חנה, כרכור, זכרון ובנימינה','שומרון': 'השומרון',
        'לוד': 'רמלה ולוד','אילת': 'אילת והערבה','הערבה': 'אילת והערבה'}

    def map_area(area):
        area = str(area)  # Convert to string
        for key, value in area_mapping.items():
            if key in area:
                return value
        return 'אחר'

    df['Processed_Area'] = df[area_column].apply(map_area)
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def convert_to_date_for_age(df, date_column_name):
    """Function Explanation: convert_to_date_for_age
        The convert_to_date_for_age function processes a date column in a DataFrame
        to calculate the age of a car based on its publication date."""
    def try_parse_date(value):
        try:
            # Try parsing the value as a date
            parsed_date = pd.to_datetime(value, format='%d/%m/%Y', errors='coerce')
            if not pd.isnull(parsed_date):
                return parsed_date
            else:
                return np.nan
        except (ValueError, TypeError):
            return np.nan

    # Apply the function to the date column and create a new column with the results
    df['Processed_Repub_date_n'] = df[date_column_name].apply(try_parse_date)
    
    # Calculate the age of the car based on the new date column
    df['Age_Car_Repub_date'] = df['Processed_Repub_date_n'].dt.year - df['Year']
    
    # Handle cases where the date is NaN
    df.loc[pd.isnull(df['Processed_Repub_date_n']), 'Age_Car_Repub_date'] = np.nan
    
    # Drop the Processed_Repub_date_n column
    df.drop(columns=['Processed_Repub_date_n'], inplace=True)
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------

def ownership_least_frequent(df, column_name, new_column_name):
    """Function Explanation: ownership_least_frequent
        The ownership_least_frequent function processes a DataFrame column to handle less frequent
        and undefined categories, specifically for ownership information. 
        It also assigns a specific value based on a condition."""
    
    # Define a function to merge similar categories
    def merge_similar_categories(value):
        if isinstance(value, str):
            if 'לא מוגדר' in value or 'None' in value:
                return np.nan  # Return NaN for undefined categories
        return value
    
    # Apply the merging function to the new column
    df[new_column_name] = df[column_name].apply(merge_similar_categories)    
    if column_name == 'Prev_ownership':
        df.loc[df['Hand'] == 1, new_column_name] = 'חדש'

    return df
