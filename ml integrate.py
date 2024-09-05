from flask import Flask, request, jsonify
import joblib  # to load the ML model
import numpy as np


app = Flask(__name__)

# Load your ML model (assumes a pre-trained model is saved as model.pkl)
model=joblib.load('simsforecasting.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the web page form (sent as JSON)
    data = request.get_json()

    # Extract features from the data
    product = data['product']
    quantity = data['quantity']

    # For example, convert input into a suitable format for the ML model
    # feature_vector = transform_data(product, quantity)
    feature_vector = np.array([quantity])  # Simple example, adjust based on your model

    # Predict using the ML model
    prediction = model.predict([feature_vector])

    # Return the prediction result to the front-end
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
