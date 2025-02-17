from django.shortcuts import render
from django.shortcuts import render,redirect
from django.contrib import messages
import urllib.request
import urllib.parse

from django.conf import settings
from django.contrib.auth import authenticate, login

def adminlogin(req):
    if req.method == 'POST':
        username = req.POST.get('username')
        password = req.POST.get('password')
        print("hello")
        print(username,password)
        # Check if the provided credentials match
        if username == 'admin' and password   == 'admin':
            messages.success(req, 'You are logged in.')
            return redirect('adashboard')  # Redirect to the admin dashboard page
        else:
             messages.error(req, 'You are trying to log in with wrong details.')
             return redirect('adashboard')  # Redirect to the login page (named 'admin' here)

    # Render the login page if the request method is GET
    return render(req, 'main/adminlogin.html')

# Create your views here.


def adashboard(req):
    return render(req, 'admin/adashboard.html')

def upload(req):
    return render(req, 'admin/upload.html')


from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from .models import TextCNNMetrics  # Assuming you are using a model to store metrics

def cnn(request):
    # Load and preprocess your CSV file dataset
    try:
        data = pd.read_csv('dataset.csv', encoding='latin1')  # Change encoding if needed
    except Exception as e:
        return HttpResponse(f"Error reading the CSV file: {e}")

    # Print the columns for debugging
    print("Columns in the DataFrame:", data.columns.tolist())  # Print columns

    # Clean up DataFrame by dropping unnecessary columns
    data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)

    # Rename the 'Patient Response (Y/N)' column to 'Patient Response'
    data.rename(columns={'Patient Response (Y/N)': 'Patient Response'}, inplace=True)

    # Check for the 'Patient Response' column
    if 'Patient Response' not in data.columns:
        print("Warning: 'Patient Response' column not found in the data.")
        return HttpResponse("Error: 'Patient Response' column not found in the data.")

    # Encode categorical columns
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Disease Type'] = label_encoder.fit_transform(data['Disease Type'])
    data['Drug Name'] = label_encoder.fit_transform(data['Drug Name'])
    data['Patient Response'] = label_encoder.fit_transform(data['Patient Response'])  # Y/N to 1/0

    # Convert numerical columns to numeric, coercing errors to NaN
    numeric_columns = ['Age', 'Dosage (mg)', 'Treatment Duration (days)', 'Drug Efficacy (%)']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set errors to NaN

    # Drop rows with NaN values (if any)
    data.dropna(subset=numeric_columns, inplace=True)

    # Scale numerical columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Split features and target
    X = data.drop(['Patient ID', 'Patient Response'], axis=1)  # Drop irrelevant columns
    y = data['Patient Response']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Predictions and metrics calculation
    y_pred = model.predict(X_test).round()
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # Save metrics to the database
    cnn_metrics = TextCNNMetrics(accuracy=accuracy, auc=auc)
    cnn_metrics.save()

    # Fetch the latest metrics for display
    metrics = TextCNNMetrics.objects.latest('id')

    # Render the metrics in the template
    return render(request, 'admin/cnn.html', {'metrics': metrics})
