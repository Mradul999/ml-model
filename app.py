from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import pickle
import os

# Define paths for saving and loading the model and encoders
model_dir = 'models'
model_save_path = os.path.join(model_dir, 'rf_model.pkl')
encoder_save_path = os.path.join(model_dir, 'label_encoders.pkl')

# Load the trained model and encoders
with open(model_save_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_save_path, 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Flask web app
app = Flask(__name__)

# Configure the SQLite database, relative to the app instance folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the KeyValue model (not used directly in prediction, but as an example of saving data)
class KeyValue(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), nullable=False)
    value = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<KeyValue {self.key}={self.value}>"

# Create the database and the table
with app.app_context():
    db.create_all()

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Function to handle unseen labels in categorical columns
def handle_unseen_labels(encoder, label):
    if label not in encoder.classes_:
        # Handle unseen label by appending it to the classes_ array
        new_classes = np.append(encoder.classes_, label)
        encoder.classes_ = new_classes
    return encoder.transform([label])[0]

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        data = request.form.to_dict()
        
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess the input data
        for column in ['Crop', 'Season', 'State']:
            input_data[column] = input_data[column].apply(lambda x: handle_unseen_labels(label_encoders[column], x))

        # Convert input features to numeric values
        numeric_features = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_data[numeric_features] = input_data[numeric_features].apply(pd.to_numeric)

        # Predict using the loaded model
        prediction = model.predict(input_data)
        
        # Round the prediction to a reasonable number of decimal places
        predicted_yield = round(prediction[0], 2)

        # Display the prediction on the page
        return render_template('index.html', prediction_text=f'Predicted Crop Yield: {predicted_yield} tons')

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(debug=True)
