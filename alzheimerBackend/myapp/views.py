#views.py

from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
 

def hello(request):
    return JsonResponse({'message': 'Hello from the backend!'})

# Get the current directory of this file
current_dir = os.path.dirname(__file__)

# Load the model and scaler
model_path = os.path.join(current_dir, 'gb_model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(f"Received data: {data}")
            feature_names = [
                'FunctionalAssessment', 'ADL', 'MMSE', 'MemoryComplaints',
                'BehavioralProblems', 'SleepQuality', 'DietQuality', 'CholesterolHDL',
                'CholesterolTriglycerides', 'CholesterolTotal', 'PhysicalActivity',
                'AlcoholConsumption', 'BMI', 'CholesterolLDL', 'BMI_SystolicBP_Ratio',
                'SystolicBP', 'Age', 'DiastolicBP'
            ]
            # Create DataFrame
            input_data = pd.DataFrame([data], columns=feature_names)
            print(f"Input DataFrame: {input_data}")
            # Scale input data
            scaled_data = scaler.transform(input_data)
            print(f"Scaled data: {scaled_data}")
            # Make prediction
            prediction = model.predict(scaled_data)
            # Convert prediction to Python int
            prediction_int = int(prediction[0])
            print(f"Prediction: {prediction_int}")
            # Return prediction as JSON
            return JsonResponse({'prediction': prediction_int})
        except Exception as e:
            print(f"Error processing prediction: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)