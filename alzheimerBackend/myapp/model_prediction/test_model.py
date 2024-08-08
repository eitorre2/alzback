import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('gb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a sample input for prediction
sample_input = np.array([[30, 5, 25, 1, 2, 3, 4, 50, 150, 200, 3, 2, 22, 100, 0.5, 120, 65, 80]])

# Scale the input
scaled_input = scaler.transform(sample_input)

# Make a prediction
prediction = model.predict(scaled_input)

print(f"Prediction: {prediction[0]}")
