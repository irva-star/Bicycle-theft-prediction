import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# ---------------------------------------------------------
#                   DATA CLEANING
# ---------------------------------------------------------

# Get the current working directory
directory = os.getcwd()

# Modify the directory to get the file Bicycle_Thefts_Open_Data.csv (same directory as the notebook)
file_path = os.path.join(directory, 'python_server/Bicycle_Thefts_Open_Data.csv')

# Load the dataset
bike_data = pd.read_csv(file_path)

# Keep only the relevant columns
relevant_columns = [
   'OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'DIVISION', 'LOCATION_TYPE',
    'PREMISES_TYPE', 'BIKE_TYPE', 'BIKE_COST', 'STATUS',
    'NEIGHBOURHOOD_158'
]

filtered_bike_data = bike_data[relevant_columns]

# Save the modified dataset
filtered_file_path = os.path.join(directory, 'Bicycle_Thefts_Filtered.csv')
filtered_bike_data.to_csv(filtered_file_path, index=False)

bike_dataf = pd.read_csv(filtered_file_path)
bike_dataf.head().T

bike_dataf.isnull().sum()
print("Unique values in 'STATUS' before mapping:", bike_dataf['STATUS'].unique())

# Fill numerical columns with their respective mean values
numerical_columns = bike_dataf.select_dtypes(include=['int64', 'float64']).columns
bike_dataf[numerical_columns] = filtered_bike_data[numerical_columns].fillna(
    bike_dataf[numerical_columns].mean()
)

# Save the updated dataset
filled_numerical_file_path = os.path.join(directory,'Bicycle_Thefts_Filled_Numerical.csv')
filtered_bike_data.to_csv(filled_numerical_file_path, index=False)

filled_numerical_file_path

# Remove rows where Status is 'UNKNOWN'
bike_dataf = bike_dataf[bike_dataf['STATUS'].fillna('').str.upper() != 'UNKNOWN']

# Fill the "BIKE_COST" column with its mean
# Fill numerical columns with their respective mean values
numerical_columns = bike_dataf.select_dtypes(include=['int64', 'float64']).columns
bike_dataf[numerical_columns] = filtered_bike_data[numerical_columns].fillna(
    bike_dataf[numerical_columns].mean()
)

# Save the updated dataset
filtered_bike_data.to_csv(filled_numerical_file_path, index=False)

filled_numerical_file_path
bike_dataf.head().T


from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
#                   DATA WRANGLING
# ---------------------------------------------------------
# Map STATUS values to numeric labels
status_mapping = {'STOLEN': 0, 'RECOVERED': 1}
bike_dataf['STATUS'] = bike_dataf['STATUS'].map(status_mapping)

# Verify the mapping
print("Updated STATUS column value counts:")
print(bike_dataf['STATUS'].value_counts())

# Encode other categorical features in the dataset
encoded_bike_dataf = bike_dataf.copy()

# Identify categorical columns
categorical_columns = encoded_bike_dataf.select_dtypes(include=['object']).columns

# Apply label encoding to all categorical columns
label_encoders = {}
for col in categorical_columns:
    encoder = LabelEncoder()
    encoded_bike_dataf[col] = encoder.fit_transform(encoded_bike_dataf[col].astype(str))
    label_encoders[col] = encoder  # Store the encoder for potential inverse transformation

# Verify the encoded dataset
encoded_bike_dataf.head()


from sklearn.preprocessing import StandardScaler

# Identify numerical columns in the encoded dataset
numerical_columns = encoded_bike_dataf.select_dtypes(include=['int64']).columns.drop('STATUS')

# Apply Standard Scaling to numerical columns
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(encoded_bike_dataf[numerical_columns])

# Create a DataFrame for scaled numerical data
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns, index=encoded_bike_dataf.index)

# Replace numerical columns in the original dataset with scaled values
scaled_bike_dataf = encoded_bike_dataf.copy()
scaled_bike_dataf[numerical_columns] = scaled_numerical_df

# Verify the scaled dataset
scaled_bike_dataf.head()
# Check unique values in STATUS column
print("Unique values in 'STATUS':", scaled_bike_dataf['STATUS'].unique())

# ---------------------------------------------------------
#                   MODEL TRAINING
# ---------------------------------------------------------

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Separate features and target
X = scaled_bike_dataf.drop('STATUS', axis=1)
y = scaled_bike_dataf['STATUS'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure target column is integer
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Train a model on undersampled data
model_under = RandomForestClassifier(random_state=42)
model_under.fit(X_train_under, y_train_under)

# Predict and evaluate for undersampling
y_pred_under = model_under.predict(X_test)
print("\n=== Undersampling (Majority Class Reduced) ===")
print("Classification Report:\n", classification_report(y_test, y_pred_under))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_under))
print("ROC AUC Score:", roc_auc_score(y_test, model_under.predict_proba(X_test)[:, 1]))



from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# API Routes
# Get options of the location names in the dataset ,the bike_type,the bike cost ,premises, neighbourhood
@app.route('/options', methods=['GET'])
def get_options():
    options = {
        'DIVISION': bike_data['DIVISION'].unique().tolist(),
        'LOCATION_TYPE': bike_data['LOCATION_TYPE'].unique().tolist(),
        'PREMISES_TYPE': bike_data['PREMISES_TYPE'].unique().tolist(),
        'NEIGHBOURHOOD_158': bike_data['NEIGHBOURHOOD_158'].unique().tolist(),
        'BIKE_TYPE': bike_data['BIKE_TYPE'].unique().tolist(),
    }
    response = jsonify(options)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Predict the status of a bike theft
@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting")

    # Get the input data
    data = request.get_json()
    print(data)

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Preprocess the input data
    input_data = preprocessor.preprocess(input_data)

    # Make predictions
    prediction = model_under.predict(input_data)

    return jsonify({'prediction': prediction[0]})
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)