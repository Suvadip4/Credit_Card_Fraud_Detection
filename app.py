# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features =request.form.get('details')  # Replace 'input_field' with the name of your input field
    values =int_features.split(',')
    final_features = [np.array(values,dtype=np.float64)]
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Fradulent Transaction' if prediction[0] == 1 else 'Legit Transaction'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)