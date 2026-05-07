from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
import joblib
import numpy as np
from .forms import CustomUserCreationForm

# Load the models
random_forest_model = joblib.load('emotion_app/random_forest_emotion_model.pkl')
tfidf_vectorizer = joblib.load('emotion_app/tfidf_vectorizer.pkl')

# Home page view
def home(request):
    return render(request, 'home.html')

# Register page view
def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

# Login page view
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('prediction')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

# Logout view
def logout_view(request):
    logout(request)
    return redirect('home')

# Prediction page view (requires login)
@login_required
def prediction(request):
    prediction_result = None  # Initialize the variable that will hold the result of the prediction
    output=""
    confidence_score=""
    if request.method == 'POST':
        user_input = request.POST.get('text_input', '').strip()
        if user_input:
            # Apply TF-IDF transformation and make prediction
            transformed_input = tfidf_vectorizer.transform([user_input])
            probabilities  = random_forest_model.predict_proba(transformed_input)[0]  # Get the first (and only) prediction
            prediction_result = np.argmax(probabilities)
            confidence_score = probabilities[prediction_result] * 100  # Confidence score as percentage


            # print(type(prediction_result))
            if prediction_result==0:
                output="sadness"
            elif prediction_result==1:
                output="joy"
            elif prediction_result==2:
                output="love"
            elif prediction_result==3:
                output="anger"
            elif prediction_result==4:
                output="fear"
            else:
                output="Ego"            

                
    return render(request, 'prediction.html', {'prediction': output, 'confidence_score': confidence_score})
