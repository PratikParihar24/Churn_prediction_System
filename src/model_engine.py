import joblib
import os

def save_model_artifacts(model, scaler, model_path='models/churn_rf_model.pkl', scaler_path='models/scaler.pkl'):
    """Saves the trained model and scaler to the models directory."""
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model and Scaler saved successfully in /models!")

# In your notebook, you can now call:
# from src.model_engine import save_model_artifacts
# save_model_artifacts(rf_model, scaler)