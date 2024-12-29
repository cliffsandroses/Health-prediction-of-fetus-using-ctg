import sys
import importlib.metadata as pkg_resources
import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# Global features list
features = [
    "baseline_value", "accelerations", "fetal_movement", 
    "uterine_contractions", "light_decelerations", 
    "severe_decelerations", "prolongued_decelerations", 
    "abnormal_short_term_variability", 
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max",
    "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

def load_models():
    try:
        models = {
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_model.pkl'),
            'SVM': joblib.load('models/svm_model.pkl'),
            'Neural Network': tf.keras.models.load_model('models/neural_network_model.keras')
        }
        scaler = joblib.load('models/data_scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def validate_input(input_data, features=features):
    if len(input_data) != len(features):
        st.error(f"Please provide all {len(features)} features")
        return False
    if any(value is None for value in input_data):
        st.error("All input fields must be filled")
        return False
    return True

def predict_fetal_health(input_data, model, scaler, model_type):
    input_scaled = scaler.transform([input_data])
    
    health_categories = {0: "Normal", 1: "Suspicious", 2: "Pathological"}
    
    if model_type in ['Random Forest', 'Gradient Boosting', 'SVM']:
        prediction = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]
    else:  # Neural Network
        proba = model.predict(input_scaled)[0]
        prediction = np.argmax(proba) + 1
    
    return health_categories[prediction], proba

def main():
    st.title("Fetal Health Prediction")
    
    # Load models
    models, scaler = load_models()
    if not models or not scaler:
        st.error("Failed to load models")
        return
    
    # Feature input
    st.sidebar.header("CTG Data Input")
    input_data = []
    for feature in features:
        value = st.sidebar.number_input(
            feature.replace('_', ' ').title(), 
            step=0.1, 
            format="%.2f"
        )
        input_data.append(value)
    
    # Model selection
    model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
    
    if st.button("Predict Fetal Health"):
        if validate_input(input_data):
            prediction, probabilities = predict_fetal_health(
                input_data, 
                models[model_choice], 
                scaler, 
                model_choice
            )
            
            st.write(f"**Predicted Fetal Health:** {prediction}")
            st.write("**Prediction Probabilities:**")
            st.bar_chart(probabilities)

if __name__ == "__main__":
    main()
